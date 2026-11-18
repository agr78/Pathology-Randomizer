"""Manifold recall in the 8D content-descriptor space, at large synthetic n.

Companion to estimate_content_coverage.py, which does the full per-axis /
kNN / recall breakdown at n~60. This one does ONE thing at scale: recall
against thousands of synthetic samples, with checkpointing so a long run can
resume, plus PCA and triplet-animation views of the enclosure.

Sample size is the reason it exists. At n=60 the recall estimate is unstable
-- measured on BraTS across three seeds, k=5 recall ran 41.7-81.7% with no
code change -- because the k-th nearest neighbour radius is itself noisy.
Thousands of synthetic samples make the manifold estimate meaningful.

Descriptors and the recall definition are imported from
estimate_content_coverage rather than reimplemented, so the two scripts cannot
drift apart.

Usage:
    python estimate_manifold_recall.py --dataset BraTS2023-GLI --n_syn 5000
    python estimate_manifold_recall.py --dataset ATLAS --mode auto --n_syn 5000
"""
import argparse
import itertools
import os
import sys

import nibabel as nib
import numpy as np
import torch

from eval_lesion_dice import DATASETS, dataset_key, list_cases, TARGET_LABELS
from crdmw_generator import paired_pathology_generator
from util import make_mode_sampler
# Single source of truth for the descriptors, the mode matching and the recall
# definition. Reimplementing any of them here is how the two scripts would
# start disagreeing about what they measure.
from estimate_content_coverage import (content_descriptors, DESCRIPTORS, MODES,
                                       DATASET_MODES, LABEL_LIST_SEGMENTATION,
                                       LESION_COMPACT_IDX, manifold_membership)

DEFAULT_OUT = "./manifold"


def log(msg):
    """Print and flush so output appears in SLURM logs as it happens."""
    print(msg)
    sys.stdout.flush()


def pca_project(syn, real, n_components=2):
    """PCA fitted on SYNTHETIC only.

    The question is where real points fall relative to the synthetic cloud, so
    the basis must not be defined partly by the real data it is used to judge.
    """
    mu = syn.mean(axis=0)
    Sc, Rc = syn - mu, real - mu
    cov = (Sc.T @ Sc) / max(Sc.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    V = evecs[:, order[:n_components]]
    frac = evals[order[:n_components]] / max(evals.sum(), 1e-12)
    return Sc @ V, Rc @ V, frac


def create_enclosure_animation(Rz, Sz, dataset_name, save_path):
    """All C(8,3) = 56 descriptor triplets, one frame each.

    A 2D PCA of eight roughly-independent descriptors shows only ~35% of the
    variance, so no single projection is faithful. Stepping through every
    triplet shows the enclosure from each combination of raw axes instead.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers 3d
    except ImportError:
        log("matplotlib not available; skipping animation.")
        return

    combos = list(itertools.combinations(range(len(DESCRIPTORS)), 3))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        i, j, k = combos[frame]
        ax.scatter(Sz[:, i], Sz[:, j], Sz[:, k], c="tab:blue", alpha=0.05, s=2,
                   label="synthetic")
        ax.scatter(Rz[:, i], Rz[:, j], Rz[:, k], c="tab:red", alpha=0.7, s=15,
                   label="real")
        ax.set_xlabel(DESCRIPTORS[i], fontsize=8)
        ax.set_ylabel(DESCRIPTORS[j], fontsize=8)
        ax.set_zlabel(DESCRIPTORS[k], fontsize=8)
        ax.set_title(f"enclosure triplet {frame+1}/{len(combos)}: {dataset_name}",
                     fontsize=10)
        for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            lim([-4, 4])

    log(f"rendering {len(combos)}-frame animation...")
    FuncAnimation(fig, update, frames=len(combos), interval=200).save(
        save_path, writer="pillow")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="BraTS2023-GLI")
    ap.add_argument("--data_root", default=os.environ.get('PR_EVAL_DATA_ROOT', '/path/to/eval_data_root'))
    ap.add_argument("--syn_data_dir",
                    default=os.environ.get('PR_DATA_DIR', '/path/to/data_dir'))
    ap.add_argument("--n_real", type=int, default=100)
    ap.add_argument("--n_syn", type=int, default=5000)
    ap.add_argument("--target", default="whole", choices=sorted(TARGET_LABELS))
    ap.add_argument("--mode", default="auto",
                    help="auto (matched to the dataset), mixture (the deployed setting), or "
                         "one of " + "/".join(MODES) + ". Comparing a specific real pathology "
                         "against the full mixture asks the wrong question -- a mixture of five "
                         "morphologies is trivially distinguishable from any one of them.")
    ap.add_argument("--couple_prob", type=float, default=None,
                    help="Probability each NON-primary mode is additionally activated. Defaults "
                         "to 0.0 when --mode names a single mode (so the measurement "
                         "characterises that mode) and 0.15 for mixture.")
    ap.add_argument("--k", type=int, default=3, help="k for the kNN manifold radii.")
    ap.add_argument("--no_animation", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    args = ap.parse_args()

    root_out = os.path.expanduser(args.out_dir)
    ckpt_dir = os.path.join(root_out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.mode == "mixture":
        mode_label, weights = "mixture (all modes)", (0.20, 0.15, 0.15, 0.25, 0.25)
    elif args.mode == "auto":
        mode_label, weights = DATASET_MODES.get(
            dataset_key(args.dataset), ("mixture (all modes)", (0.20, 0.15, 0.15, 0.25, 0.25)))
        mode_label = f"{mode_label} (auto-matched)"
    else:
        if args.mode not in MODES:
            raise SystemExit(f"--mode must be auto/mixture/{'/'.join(MODES)}")
        w = [0.0] * 5
        w[MODES.index(args.mode)] = 1.0
        mode_label, weights = args.mode, tuple(w)
    couple_prob = args.couple_prob if args.couple_prob is not None else (
        0.15 if args.mode == "mixture" else 0.0)

    # Mode is part of the checkpoint identity: samples drawn under one mode are
    # not interchangeable with another's, and silently resuming across a mode
    # change would mix them.
    tag = f"{args.dataset}_{args.mode}_cp{couple_prob}"
    ckpt_file = os.path.join(ckpt_dir, f"{tag}.npz")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log(f"dataset {args.dataset} | mode {mode_label} | couple_prob {couple_prob} | "
        f"device {device}")

    real, syn = [], []
    if os.path.exists(ckpt_file):
        ck = np.load(ckpt_file, allow_pickle=True)
        real, syn = ck["real"].tolist(), ck["syn"].tolist()
        log(f"resumed from {ckpt_file}: {len(real)} real, {len(syn)} synthetic")

    if len(real) < args.n_real:
        data_dir = os.path.join(args.data_root, args.dataset)
        cases = list_cases(data_dir, args.dataset)
        if not cases:
            raise SystemExit(f"no ground-truth masks under {data_dir}")
        key = dataset_key(args.dataset)
        wanted = DATASETS[key]["labels"] or TARGET_LABELS[args.target]
        log(f"reading real masks (target {args.n_real}, labels {wanted})...")
        for idx in rng.permutation(len(cases)):
            if len(real) >= args.n_real:
                break
            _sub, _t1, gtp = cases[idx]
            gt = nib.load(gtp).get_fdata().astype(np.int32)
            while gt.ndim > 3:
                gt = gt[..., 0]
            d = content_descriptors(np.isin(gt, wanted))
            if d:
                real.append([d[k] for k in DESCRIPTORS])
                if len(real) % 20 == 0:
                    log(f"  real {len(real)}/{args.n_real}")
        np.savez(ckpt_file, real=np.array(real), syn=np.array(syn))

    if len(syn) < args.n_syn:
        log(f"generating synthetic samples (target {args.n_syn})...")
        gen = paired_pathology_generator(
            args.syn_data_dir, make_mode_sampler(restricted=False),
            label_list_segmentation=LABEL_LIST_SEGMENTATION,
            n_neutral_labels=20, label_list_normalization=[1, 18], sides="both",
            labels_to_kill=[45], list_to_kill_photo=[], list_to_kill_photo_inc_ventricles=[],
            flipping=True, mix_real_images=False, generate_pathology=True, minimal_io=True,
            # Indices, not the 56-channel one-hot: this only ever argmaxes, and
            # the one-hot is 0.92 GB per stream at 160^3 against 0.004 GB here.
            compact_labels=True,
            mode_weights=weights, couple_prob=couple_prob,
            random_kill_labels=list(range(34, 47)), random_kill_prob=0.5,
            random_kill_frac=(1.0, 1.0), device=device)

        mode_counts = {}
        while len(syn) < args.n_syn:
            case = next(gen)
            if case["is_cheap_repeat"]:
                continue
            m = (case["seg_path_idx"] == LESION_COMPACT_IDX).cpu().numpy()
            d = content_descriptors(m)
            if d:
                syn.append([d[k] for k in DESCRIPTORS])
                nm = (case.get("pathology_metadata") or {}).get("mode_name", "?")
                mode_counts[nm] = mode_counts.get(nm, 0) + 1
                if len(syn) % 100 == 0:
                    log(f"  synthetic {len(syn)}/{args.n_syn} (checkpointing)")
                    np.savez(ckpt_file, real=np.array(real), syn=np.array(syn))
        np.savez(ckpt_file, real=np.array(real), syn=np.array(syn))
        log("mode composition (confirms isolation held):")
        for nm, c in sorted(mode_counts.items(), key=lambda kv: -kv[1])[:8]:
            log(f"  {c:5d}  {nm}")

    # The generator yields from inside `with torch.no_grad():`; grad mode is
    # thread-local global state and stays off after next() returns.
    torch.set_grad_enabled(True)

    R, S = np.array(real), np.array(syn)
    if R.shape[0] < 5 or S.shape[0] < 5:
        raise SystemExit(f"too few usable samples (real {R.shape[0]}, syn {S.shape[0]})")

    mu, sd = S.mean(0), S.std(0) + 1e-9
    Rz, Sz = (R - mu) / sd, (S - mu) / sd

    np.savez(os.path.join(root_out, f"manifold_data_{tag}.npz"),
             Rz=Rz, Sz=Sz, descriptors=np.array(DESCRIPTORS), mode=mode_label)

    log("\n" + "=" * 72)
    log(f"RESULT: {args.dataset}  [{mode_label}]")
    log("=" * 72)
    log(f"  {R.shape[0]} real vs {S.shape[0]} synthetic, {len(DESCRIPTORS)}D descriptor space")
    for k in (args.k, args.k + 2):
        rec = manifold_membership(Rz, Sz, k=k).mean()
        log(f"  k={k}  manifold recall {rec:6.1%}   (real samples inside the "
            f"SYNTHETIC manifold)")
    recall = manifold_membership(Rz, Sz, k=args.k).mean()
    log("")
    log("  Recall is the coverage claim: support universality is exactly high recall.")
    log("  Precision is not reported here -- it measures fidelity, which this")
    log("  generator deliberately sacrifices, and it is not what the bound needs.")
    log("=" * 72)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Sp, Rp, frac = pca_project(Sz, Rz)
        plt.figure(figsize=(10, 7))
        plt.scatter(Sp[:, 0], Sp[:, 1], c="tab:blue", alpha=0.2, s=5, label="synthetic")
        plt.scatter(Rp[:, 0], Rp[:, 1], c="tab:red", alpha=0.6, s=20,
                    label=f"real ({args.dataset})")
        plt.xlabel(f"PC1 ({frac[0]:.0%} of synthetic variance)")
        plt.ylabel(f"PC2 ({frac[1]:.0%})")
        plt.title(f"{args.dataset} | {mode_label} | recall {recall:.1%}")
        plt.legend()
        path = os.path.join(root_out, f"recall_pca_{tag}.png")
        plt.savefig(path, dpi=130)
        plt.close()
        log(f"wrote {path}")
    except ImportError:
        log("matplotlib not available; skipping PCA plot.")

    if not args.no_animation:
        path = os.path.join(root_out, f"manifold_enclosure_{tag}.gif")
        create_enclosure_animation(Rz, Sz, args.dataset, path)
        log(f"wrote {path}")


if __name__ == "__main__":
    main()