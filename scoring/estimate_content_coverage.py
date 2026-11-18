"""Estimate coverage of real pathology CONTENT by the synthetic generator.

Factorize the joint distribution as

    P(x, y) = P(y) P(x | y)

with y the content -- the anatomical and pathological configuration, i.e. the
label map and lesion geometry -- and P(x | y) the appearance rendering.
Coverage then splits into two independent claims:

  appearance   SynthSeg's contribution. The GMM draws mu ~ U(25,225) per label
               with no tie to real tissue values, so the synthetic contrast
               distribution is far wider than any real sequence. This is the
               empirically validated part and is not what this script measures.

  content      This generator's contribution, and what is measured here.

Measuring coverage on images does not work: a discriminator separates real
from synthetic on appearance tells -- GMM-flat tissue interiors, the noise
model, resolution banding -- long before anything about pathology. Measured
AUC was 1.000 on both BraTS and ATLAS, which says the appearance model is
unrealistic (by design) and says nothing about whether real lesion geometry
falls inside the synthetic support.

Content descriptors are also the right level for a task-agnostic claim. The
content is what every downstream task shares: segmentation predicts it,
reconstruction and super-resolution are conditioned on it. It is not a
segmentation-specific object.

Three measures, deliberately not relying on classifier calibration:

  per-axis coverage   fraction of real samples inside the synthetic range on
                      each descriptor. Directly checks support inclusion.
  kNN support ratio   distance from each real sample to its nearest synthetic
                      neighbour, relative to typical synthetic-synthetic
                      spacing. >> 1 means real content sits outside the cloud.
  AUC                 separability on descriptors alone. Here it is meaningful
                      because appearance has been factored out.

Usage:
    python estimate_content_coverage.py --dataset BraTS2023-GLI --n_real 60 --n_syn 60
"""
import argparse
import os
from functools import lru_cache

import numpy as np
import nibabel as nib
import torch

from eval_lesion_dice import DATASETS, dataset_key, list_cases, TARGET_LABELS
from crdmw_generator import paired_pathology_generator
from util import make_mode_sampler

LABEL_LIST_SEGMENTATION = [0, 11, 12, 13, 16, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                           43, 44, 46, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 17, 47, 49, 51,
                           53, 55, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 48, 50,
                           52, 54, 56]
LESION_COMPACT_IDX = LABEL_LIST_SEGMENTATION.index(33)

# mode_weights order, from mcp_pathology_randomizer.py
MODES = ["Local", "Global", "Atrophy", "Vascular", "Tract"]

# Which mode each dataset's pathology corresponds to. Comparing a specific
# real pathology against the full synthetic MIXTURE asks the wrong question --
# a mixture of five morphologies is trivially distinguishable from any one of
# them regardless of whether the relevant mode covers it. The question that
# matters is whether the RELEVANT mode reaches the real content.
DATASET_MODES = {
    "BraTS": ("Local", (1.0, 0.0, 0.0, 0.0, 0.0)),      # compact mass + mass effect
    "ATLAS": ("Vascular", (0.0, 0.0, 0.0, 1.0, 0.0)),   # territorial / lacunar infarct
    "WMH":   ("Tract", (0.0, 0.0, 0.0, 0.0, 1.0)),      # multifocal periventricular
}

DESCRIPTORS = ["log_volume", "n_components", "largest_frac", "compactness",
               "surface_to_volume", "elongation", "flatness", "rel_eccentricity"]


@lru_cache(maxsize=1024)
def _voxelized_sphere_surface(n):
    """Boundary-voxel count of a VOXELIZED ball holding ~n voxels.

    surface_to_volume previously divided a discrete 6-neighbour boundary count
    by the CONTINUOUS equivalent-sphere area, which mixes units: a small object
    is coarsely discretised and undercounts its boundary, while a large one
    converges toward the continuous value. Measured on exact spheres with the
    shape held constant, the ratio drifted 0.696 (r=4) to 0.810 (r=25), giving
    a spurious +0.96 correlation with log_volume.

    That contaminated every surface_to_volume number: synthetic sat at 0.81-0.89
    against real BraTS at +0.08, and the apparent difference was largely this
    artifact rather than morphology.

    Normalising against a voxelized ball of the same volume removes it -- the
    reference is discretised the same way the measurement is. Spheres now score
    exactly 1.000 at every size. The scale is also more readable: 1.0 means "as
    smooth as a ball of this volume", above 1.0 means raggeder.

    Cached because it rebuilds a ball per distinct volume; volumes repeat
    heavily across a coverage run.
    """
    r = (3.0 * n / (4.0 * np.pi)) ** (1.0 / 3.0)
    D = int(2 * r + 6)
    zz, yy, xx = np.meshgrid(*[np.arange(D)] * 3, indexing="ij")
    c = (D - 1) / 2.0
    m = ((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2) <= r ** 2
    from scipy import ndimage as _nd
    return max(int(m.sum() - _nd.binary_erosion(m).sum()), 1)


def content_descriptors(mask):
    """Shape descriptors of a binary lesion mask, in units that do not depend
    on grid size or voxel count, so real and synthetic are comparable despite
    living on different grids.

    Returns None when the mask is empty or too small for shape statistics to
    mean anything.
    """
    n = int(mask.sum())
    if n < 27:
        return None

    from scipy import ndimage
    lab, ncomp = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())[1:]
    largest = sizes.max() / n

    idx = np.argwhere(mask).astype(np.float64)
    extent = idx.max(0) - idx.min(0) + 1
    compactness = n / float(np.prod(extent))

    # Surface via 6-neighbour boundary count, normalised by the surface of a
    # sphere of equal volume -- 1.0 for a perfect sphere, larger when ragged.
    er = ndimage.binary_erosion(mask)
    surf = int(mask.sum() - er.sum())
    # Normalised by a VOXELIZED ball of the same volume, not the continuous
    # sphere area -- see _voxelized_sphere_surface. 1.0 = as smooth as a ball.
    s2v = surf / _voxelized_sphere_surface(n)

    cen = idx - idx.mean(0)
    cov = (cen.T @ cen) / len(cen)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ev = np.maximum(ev, 1e-9)
    elongation = float(np.sqrt(ev[0] / ev[1]))
    flatness = float(np.sqrt(ev[1] / ev[2]))

    # Position relative to the mask's own bounding volume, so it does not
    # depend on where the head sits in the grid.
    shape = np.array(mask.shape, dtype=float)
    rel = np.abs(idx.mean(0) / shape - 0.5).sum()

    return dict(log_volume=float(np.log10(n)), n_components=float(ncomp),
                largest_frac=float(largest), compactness=float(compactness),
                surface_to_volume=float(s2v), elongation=elongation,
                flatness=flatness, rel_eccentricity=float(rel))


def manifold_membership(query, ref, k=3):
    """Kynkaanniemi et al. (2019) k-NN manifold estimate.

    The reference manifold is approximated by the union of balls centred on
    each reference sample, with radius equal to that sample's distance to its
    k-th nearest neighbour within the reference set. A query point counts as
    inside if it falls within ANY such ball.

    This is the direct enclosure test. Per-axis bracketing only checks
    marginals -- a point can sit inside every axis range while being far from
    every actual sample, in a hole the marginals cannot see. The median kNN
    ratio gives one number for the whole set with no per-sample verdict.

    Returns a boolean array over query samples.
    """
    d_rr = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=2)
    np.fill_diagonal(d_rr, np.inf)
    # k-th nearest: sort each row and take index k-1
    radii = np.sort(d_rr, axis=1)[:, k - 1]
    d_qr = np.linalg.norm(query[:, None, :] - ref[None, :, :], axis=2)
    return (d_qr <= radii[None, :]).any(axis=1)


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="BraTS2023-GLI")
    ap.add_argument("--data_root", default=os.environ.get('PR_EVAL_DATA_ROOT', '/path/to/eval_data_root'))
    ap.add_argument("--syn_data_dir",
                    default=os.environ.get('PR_DATA_DIR', '/path/to/data_dir'))
    ap.add_argument("--n_real", type=int, default=60)
    ap.add_argument("--n_syn", type=int, default=60)
    ap.add_argument("--target", default="whole", choices=sorted(TARGET_LABELS))
    ap.add_argument("--mode", default="auto",
                    help="Which generator mode to sample: auto (matched to the dataset), "
                         "mixture (all five, the deployed setting), or one of "
                         + "/".join(MODES) + ".")
    ap.add_argument("--couple_prob", type=float, default=None,
                    help="Probability each NON-primary mode is additionally activated. The "
                         "generator default is 0.15, so ~48%% of draws carry an extra mechanism "
                         "and --mode never isolates anything. Defaults to 0.0 when --mode names "
                         "a single mode (so the measurement characterises that mode), and to "
                         "0.15 for --mode mixture (the deployed setting).")
    ap.add_argument("--plot", default=None, metavar="PNG",
                    help="Write a coverage figure: per-axis bracketing, a PCA view of the "
                         "descriptor space with real points coloured by kNN support ratio, and "
                         "the ratio histogram.")
    ap.add_argument("--n_seeds", type=int, default=1,
                    help="Repeat the whole estimate with this many seeds and report mean +/- "
                         "std. At n=60 the metric is unstable: measured on BraTS across three "
                         "seeds, flatness coverage ran 55-100%% and k=5 recall 41.7-81.7%% with "
                         "no code change. Single-seed values should not be quoted.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    all_runs = []
    for _seed in range(args.seed, args.seed + args.n_seeds):
        all_runs.append(_one_estimate(args, _seed))

    if args.n_seeds > 1:
        print("\n" + "=" * 72)
        print(f"AGGREGATE over {args.n_seeds} seeds (mean +/- std)")
        print("=" * 72)
        keys = [("recall3", "manifold recall k=3"), ("recall5", "manifold recall k=5"),
                ("prec3", "precision k=3"), ("all_axes", "inside range on ALL axes"),
                ("knn", "kNN support ratio")]
        for k, label in keys:
            v = np.array([r[k] for r in all_runs])
            unit = "" if k == "knn" else "%"
            scale = 1.0 if k == "knn" else 100.0
            print(f"  {label:28s} {scale*v.mean():6.1f}{unit} +/- {scale*v.std():.1f}{unit}"
                  f"   [{scale*v.min():.1f}, {scale*v.max():.1f}]")
        print("\n  per-axis coverage:")
        for j, name in enumerate(DESCRIPTORS):
            v = np.array([r["per_axis"][j] for r in all_runs])
            print(f"    {name:20s} {100*v.mean():6.1f}% +/- {100*v.std():.1f}%"
                  f"   [{100*v.min():.0f}, {100*v.max():.0f}]")
        print("\n  Single-seed values are not stable at this sample size; quote the mean")
        print("  and spread, or raise --n_real/--n_syn.")
    return


def _one_estimate(args, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- real content, from ground-truth masks ---
    data_dir = os.path.join(args.data_root, args.dataset)
    cases = list_cases(data_dir, args.dataset)
    if not cases:
        raise SystemExit(f"no ground-truth masks under {data_dir}")
    key = dataset_key(args.dataset)
    wanted = DATASETS[key]["labels"] or TARGET_LABELS[args.target]
    picks = [cases[i] for i in rng.choice(len(cases), size=min(args.n_real, len(cases)),
                                          replace=False)]
    print(f"reading {len(picks)} real masks from {args.dataset} (labels {wanted})")

    real = []
    for i, (sub, _t1, gtp) in enumerate(picks):
        gt = nib.load(gtp).get_fdata().astype(np.int32)
        while gt.ndim > 3:
            gt = gt[..., 0]
        d = content_descriptors(np.isin(gt, wanted))
        if d is not None:
            real.append(d)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(picks)}")

    # --- synthetic content, from the generator's own lesion channel ---
    if args.mode == "mixture":
        mode_label, weights = "mixture (all modes)", (0.20, 0.15, 0.15, 0.25, 0.25)
    elif args.mode == "auto":
        mode_label, weights = DATASET_MODES.get(key, ("mixture (all modes)",
                                                      (0.20, 0.15, 0.15, 0.25, 0.25)))
        mode_label = f"{mode_label} (auto-matched to {key})"
    else:
        if args.mode not in MODES:
            raise SystemExit(f"--mode must be auto/mixture/{'/'.join(MODES)}")
        w = [0.0] * 5
        w[MODES.index(args.mode)] = 1.0
        mode_label, weights = args.mode, tuple(w)

    # Isolate by default when a single mode is requested: mode_weights alone
    # only selects the PRIMARY mode, and coupling then adds others, which is
    # what made the ATLAS Vascular run show n_components >= 83 (multifocal
    # Tract-Aligned contamination) rather than territorial infarct geometry.
    if args.couple_prob is not None:
        couple_prob = args.couple_prob
    else:
        couple_prob = 0.15 if args.mode == "mixture" else 0.0

    print(f"generating {args.n_syn} synthetic label maps  "
          f"[mode: {mode_label}, couple_prob {couple_prob}]")
    mode_counts = {}
    gen = paired_pathology_generator(
        args.syn_data_dir, make_mode_sampler(restricted=False),
        label_list_segmentation=LABEL_LIST_SEGMENTATION,
        n_neutral_labels=20, label_list_normalization=[1, 18], sides="both",
        labels_to_kill=[45], list_to_kill_photo=[], list_to_kill_photo_inc_ventricles=[],
        flipping=True, mix_real_images=False, generate_pathology=True, minimal_io=True,
        random_kill_labels=list(range(34, 47)), random_kill_prob=0.5,
        random_kill_frac=(1.0, 1.0), mode_weights=weights, couple_prob=couple_prob,
        device="cuda" if torch.cuda.is_available() else "cpu")

    syn, tries = [], 0
    while len(syn) < args.n_syn and tries < args.n_syn * 4:
        tries += 1
        case = next(gen)
        if case["is_cheap_repeat"]:
            continue
        m = (torch.argmax(case["seg_path_onehot"], dim=-1) == LESION_COMPACT_IDX).cpu().numpy()
        d = content_descriptors(m)
        if d is not None:
            syn.append(d)
            nm = (case.get("pathology_metadata") or {}).get("mode_name", "?")
            mode_counts[nm] = mode_counts.get(nm, 0) + 1
            if len(syn) % 20 == 0:
                print(f"  {len(syn)}/{args.n_syn}")

    # Generator yields from inside `with torch.no_grad():`; grad mode is global
    # thread-local state and stays off after next(gen) returns.
    torch.set_grad_enabled(True)

    if len(real) < 5 or len(syn) < 5:
        raise SystemExit(f"too few usable samples (real {len(real)}, syn {len(syn)})")

    R = np.array([[d[k] for k in DESCRIPTORS] for d in real])
    S = np.array([[d[k] for k in DESCRIPTORS] for d in syn])
    print(f"\nusable: {len(R)} real, {len(S)} synthetic")
    print("synthetic mode composition (confirms isolation actually held):")
    for nm, c in sorted(mode_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {c:3d}  {nm}")

    # --- 1. per-axis support inclusion ---
    print("\n" + "=" * 72)
    print("per-axis coverage: fraction of real samples inside the synthetic range")
    print("=" * 72)
    lo, hi = S.min(0), S.max(0)
    inside_all = np.ones(len(R), dtype=bool)
    for j, name in enumerate(DESCRIPTORS):
        ins = (R[:, j] >= lo[j]) & (R[:, j] <= hi[j])
        inside_all &= ins
        print(f"  {name:20s} {ins.mean():6.1%}   real [{R[:,j].min():8.2f}, {R[:,j].max():8.2f}]"
              f"   syn [{lo[j]:8.2f}, {hi[j]:8.2f}]")
    print(f"\n  inside synthetic range on ALL axes: {inside_all.mean():.1%}")
    print("  (a real sample outside on any axis is outside the support; this is")
    print("   necessary for coverage, not sufficient -- the joint region is smaller)")

    # --- 2. kNN support ratio ---
    mu, sd = S.mean(0), S.std(0) + 1e-9
    Rz, Sz = (R - mu) / sd, (S - mu) / sd

    def nn_dist(A, B, exclude_self=False):
        d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        if exclude_self:
            np.fill_diagonal(d, np.inf)
        return d.min(1)

    d_rs = nn_dist(Rz, Sz)
    d_ss = nn_dist(Sz, Sz, exclude_self=True)
    ratio = np.median(d_rs) / max(np.median(d_ss), 1e-9)
    print("\n" + "=" * 72)
    print("kNN support ratio (standardised descriptor space)")
    print("=" * 72)
    print(f"  median real->synthetic nearest-neighbour distance : {np.median(d_rs):.3f}")
    print(f"  median synthetic->synthetic spacing               : {np.median(d_ss):.3f}")
    print(f"  ratio                                            : {ratio:.2f}")
    print("  ~1 means real content sits among synthetic samples; >>1 means it sits")
    print("  outside the cloud and the support does not reach it")

    # --- 2b. manifold recall and precision ---
    print("\n" + "=" * 72)
    print("manifold enclosure (Kynkaanniemi et al. 2019, k-NN estimate)")
    print("=" * 72)
    for k in (3, 5):
        rec = manifold_membership(Rz, Sz, k=k).mean()
        prec = manifold_membership(Sz, Rz, k=k).mean()
        print(f"  k={k}   recall  {rec:6.1%}   (real samples inside the SYNTHETIC manifold)")
        print(f"        precision {prec:6.1%}   (synthetic samples inside the REAL manifold)")
    print()
    print("  Recall is the coverage claim: support universality is exactly high recall.")
    print("  Precision is fidelity and is intentionally low -- the generator randomises")
    print("  far beyond realism, so most synthetic samples are outside the real manifold")
    print("  by design. Low precision is not a defect here; low recall is.")

    # --- 3. per-axis gap, where coverage fails and by how much ---
    # Deliberately NOT a discriminator AUC. AUC measures whether two
    # distributions DIFFER; coverage is whether one support CONTAINS another.
    # A wide synthetic distribution containing a narrow real one is trivially
    # separable (AUC 1.0) while having perfect coverage, so AUC cannot answer
    # the question. Measured directly: per-axis bracketing near 100% alongside
    # AUC 1.000 on the same data, which is exactly that situation.
    print("\n" + "=" * 72)
    print("where coverage fails: real values outside the synthetic range")
    print("=" * 72)
    any_gap = False
    for j, name in enumerate(DESCRIPTORS):
        below = (R[:, j] < lo[j]).mean()
        above = (R[:, j] > hi[j]).mean()
        if below + above < 0.02:
            continue
        any_gap = True
        parts = []
        if below > 0.02:
            parts.append(f"{below:.0%} below {lo[j]:.2f} (real min {R[:,j].min():.2f})")
        if above > 0.02:
            parts.append(f"{above:.0%} above {hi[j]:.2f} (real max {R[:,j].max():.2f})")
        print(f"  {name:20s} " + ";  ".join(parts))
    if not any_gap:
        print("  none -- every real sample falls inside the synthetic range on every axis")

    rec3 = manifold_membership(Rz, Sz, k=3).mean()
    rec5 = manifold_membership(Rz, Sz, k=5).mean()
    prec3 = manifold_membership(Sz, Rz, k=3).mean()
    per_axis_cov = [float(((R[:, j] >= lo[j]) & (R[:, j] <= hi[j])).mean())
                    for j in range(len(DESCRIPTORS))]

    if args.plot and seed == args.seed:
        make_plot(R, S, d_rs, d_ss, lo, hi, args.plot,
                  f"{args.dataset}  |  {len(R)} real vs {len(S)} synthetic  |  mode: {mode_label}"
                  f"  |  recall {rec3:.0%}, kNN ratio {ratio:.2f}")

    print(f"\n  mode: {mode_label}")
    print("  Coverage is support inclusion, so the per-axis and kNN numbers above are")
    print("  the measurement. A failing axis names the morphology the generator cannot")
    print("  produce, which is directly actionable.")

    return dict(recall3=float(rec3), recall5=float(rec5), prec3=float(prec3),
                all_axes=float(inside_all.mean()), knn=float(ratio),
                per_axis=per_axis_cov)


def make_plot(R, S, d_rs, d_ss, lo, hi, path, title):
    """Three views of the same measurement.

    Bracketing shows the marginals; PCA shows whether real points sit inside
    the synthetic cloud jointly; the histogram shows the spread of per-sample
    support ratios rather than just their median.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(title, fontsize=12)

    # --- per-axis bracketing, normalised so all axes share one scale ---
    ax = fig.add_subplot(1, 3, 1)
    y = np.arange(len(DESCRIPTORS))
    for j in range(len(DESCRIPTORS)):
        span = max(hi[j] - lo[j], 1e-9)
        # map each axis onto [0,1] by its SYNTHETIC range, so "inside" is
        # literally "within the bar" and axes with wildly different units
        # (voxel counts vs ratios) are comparable
        sl, sh = 0.0, 1.0
        rl, rh = (R[:, j].min() - lo[j]) / span, (R[:, j].max() - lo[j]) / span
        ax.plot([sl, sh], [y[j], y[j]], lw=7, color="#c8d8e8",
                solid_capstyle="butt", zorder=1)
        covered = rl >= 0 and rh <= 1
        ax.plot([rl, rh], [y[j], y[j]], lw=3.2,
                color="#2a7d3f" if covered else "#c0392b", solid_capstyle="butt", zorder=2)
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.axvline(1, color="k", lw=0.6, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(DESCRIPTORS, fontsize=8)
    ax.set_xlabel("position within synthetic range")
    ax.set_title("per-axis bracketing\n(blue = synthetic, green/red = real in/out)", fontsize=9)
    ax.set_xlim(min(-0.15, float(((R - lo) / np.maximum(hi - lo, 1e-9)).min()) - 0.05),
                max(1.15, float(((R - lo) / np.maximum(hi - lo, 1e-9)).max()) + 0.05))
    ax.invert_yaxis()

    # --- PCA of the standardised descriptor space ---
    ax = fig.add_subplot(1, 3, 2)
    mu, sd = S.mean(0), S.std(0) + 1e-9
    Rz, Sz = (R - mu) / sd, (S - mu) / sd
    # PCA fitted on SYNTHETIC only: the question is where real points fall
    # relative to the synthetic cloud, so the basis must not be defined partly
    # by the real data it is being used to judge.
    U, sv, Vt = np.linalg.svd(Sz - Sz.mean(0), full_matrices=False)
    W = Vt[:2].T
    Sp, Rp = (Sz - Sz.mean(0)) @ W, (Rz - Sz.mean(0)) @ W
    ax.scatter(Sp[:, 0], Sp[:, 1], s=18, c="#7fa8c9", alpha=0.55,
               edgecolors="none", label="synthetic")
    sc = ax.scatter(Rp[:, 0], Rp[:, 1], s=26, c=d_rs / max(np.median(d_ss), 1e-9),
                    cmap="RdYlGn_r", vmin=0.5, vmax=5, edgecolors="k", linewidths=0.3,
                    label="real")
    plt.colorbar(sc, ax=ax, label="kNN support ratio", fraction=0.046)
    var = sv[:2] ** 2 / (sv ** 2).sum()
    ax.set_xlabel(f"PC1 ({var[0]:.0%} of synthetic variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    ax.set_title("descriptor space, PCA fitted on synthetic", fontsize=9)
    ax.legend(fontsize=8, loc="best")

    # --- distribution of support ratios ---
    ax = fig.add_subplot(1, 3, 3)
    ratios = d_rs / max(np.median(d_ss), 1e-9)
    ax.hist(ratios, bins=20, color="#7fa8c9", edgecolor="k", linewidth=0.4)
    ax.axvline(1.0, color="#2a7d3f", lw=1.5, ls="--", label="embedded (ratio 1)")
    ax.axvline(np.median(ratios), color="#c0392b", lw=1.5,
               label=f"median {np.median(ratios):.2f}")
    ax.set_xlabel("real->synthetic NN distance / synthetic spacing")
    ax.set_ylabel("real samples")
    ax.set_title("support ratio distribution", fontsize=9)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()