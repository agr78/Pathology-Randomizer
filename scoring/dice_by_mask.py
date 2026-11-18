"""Dice for each (reg map x brain mask) combination, per case.

The analysis region sets the threshold AND the scoring region, so swapping the
mask moves every number. This lays that out explicitly instead of leaving it
implicit in auc_brain's TISSUE[] default.

Same GT-free rule as eval/auc_brain.py --fixed_z:
    thr = median(|reg| in brain) + z * 1.4826 * MAD(|reg| in brain)
    bright: sreg > thr    dark: sreg < -thr    abs: |sreg| > thr
No ground truth enters the threshold. gt_outside is reported for disclosure
only -- see docs/brain-mask-choice.md.

    python eval/dice_by_mask.py --dataset BraTS2023-GLI --subjects sub-00643 \
        --derivs A B --masks synthstrip_b0 synthseg_healthy_98ea36c_synthseg_500000
"""
import argparse, os, sys
import numpy as np, nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))  # after the repo root: eval/compare_datasets.py is a stale copy
from blind_threshold import threshold_sides, cluster_mask  # noqa: E402
from compare_datasets import (EXTRACEREBRAL_COMPACT_IDX, dataset_key,  # noqa: E402
                              list_cases, output_name_for)
EX = list(EXTRACEREBRAL_COMPACT_IDX)


def auc_subsample(pos, neg, rng, n=200000):
    """The (pos, neg) subsample auc() scores; shared with prevfree_metrics() so the
    rng stream is consumed identically with and without --prevfree."""
    if pos.size == 0 or neg.size == 0:
        return None, None
    a = rng.choice(pos, min(n, pos.size), replace=False)
    b = rng.choice(neg, min(n, neg.size), replace=False)
    return a, b


def auc_from_subsample(a, b):
    if a is None:
        return float("nan")
    # Mann-Whitney: #(neg < pos) + 0.5 * #(neg == pos). The 'left' search counts strict
    # inequality; ties (right - left) get 1/2. (2026-09-28: 'right' was used for the first
    # term, which already includes ties, so ties counted 3/2 -- inflated AUC by the tied-pair
    # fraction, <=0.008 on real cases; verified against scipy.stats.mannwhitneyu.)
    sb = np.sort(b); right = np.searchsorted(sb, a, "right"); left = np.searchsorted(sb, a, "left")
    return float((left.sum() + 0.5 * (right - left).sum()) / (a.size * b.size))


def auc(pos, neg, rng, n=200000):
    return auc_from_subsample(*auc_subsample(pos, neg, rng, n))


PF_COLS = ("tpr@fpr1", "tpr@fpr5", "pauc5", "les_sens", "les_sens10", "n_fp", "lmax/p999", "D_top0.5%", "D_top1%", "D_top2%")
TOPQ = (0.005, 0.01, 0.02)   # volume-fraction cuts: keep the top q of brain voxels by score; one q for every dataset (no disease prior)
PF_MIN_VOX = 20     # GT / FP component size floor for the lesion-wise metrics


def prevfree_metrics(a, b, r_use, brain, gt, det):
    """Prevalence-free metrics for --prevfree. (a, b) = auc()'s subsample.
    tpr@fprF: TPR at the (1-F) quantile of the healthy subsample (strict >).
    pauc5: partial AUC over FPR in [0, .05] / .05 (NOT McClish); ties split .5 like auc().
    les_sens: fraction of GT components (26-conn, >= PF_MIN_VOX vox, inside brain) with >= 1
    voxel above the fixed cut `det`; les_sens10: >= 10% of its voxels. n_fp: components of
    `det` (>= PF_MIN_VOX vox) with no GT voxel. lmax/p999: max score in GT / healthy 99.9th pct."""
    from scipy import ndimage
    nan = float("nan")
    if a is None:
        tpr1 = tpr5 = pauc = nan
    else:
        bs = np.sort(b)
        tpr1 = float((a > np.quantile(bs, 0.99)).mean())
        tpr5 = float((a > np.quantile(bs, 0.95)).mean())
        fpr = 1.0 - (np.searchsorted(bs, a, "right") + np.searchsorted(bs, a, "left")) / (2.0 * bs.size)
        pauc = float(np.maximum(0.05 - fpr, 0).mean() / 0.05)   # = int_0^.05 TPR(f) df / .05
    st = np.ones((3, 3, 3), bool)
    gtb = gt & brain; det = det & brain
    lab, n = ndimage.label(gtb, structure=st)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    hit = np.bincount(lab.ravel(), weights=det.ravel(), minlength=n + 1)
    keep = sizes[1:] >= PF_MIN_VOX
    if keep.any():
        sens = float((hit[1:][keep] >= 1).mean()); sens10 = float((hit[1:][keep] >= 0.1 * sizes[1:][keep]).mean())
    else:
        sens = sens10 = nan
    lab, n = ndimage.label(det, structure=st)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    ov = np.bincount(lab.ravel(), weights=gtb.ravel(), minlength=n + 1)
    nfp = int(((sizes[1:] >= PF_MIN_VOX) & (ov[1:] == 0)).sum())
    ph, pl = r_use[brain & ~gt], r_use[gtb]
    ratio = float(pl.max() / max(np.quantile(ph, 0.999), 1e-9)) if pl.size and ph.size else nan
    rb = r_use[brain]; G = gtb.sum(); dq = []
    for q in TOPQ:   # Dice when the top q of brain voxels by score are called lesion
        t = np.quantile(rb, 1.0 - q); pred = (r_use > t) & brain
        dq.append(float(2.0 * (pred & gtb).sum() / max(pred.sum() + G, 1)))
    return (tpr1, tpr5, pauc, sens, sens10, nfp, ratio) + tuple(dq)


def average_precision(pos, neg):
    """Area under the precision-recall curve (sklearn's step-wise AP), on the FULL
    pools: unlike auc() it must not subsample, because AP depends on the
    positive:negative ratio (prevalence). Ties are broken by putting negatives
    first (conservative)."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    scores = np.concatenate([pos, neg]); lab = np.concatenate([np.ones(pos.size, bool), np.zeros(neg.size, bool)])
    order = np.lexsort((lab, -scores))          # descending score; within a tie, negatives (False) first
    lab = lab[order]
    tp = np.cumsum(lab); k = np.arange(1, lab.size + 1)
    return float((tp[lab] / k[lab]).sum() / pos.size)


NBINS = 65536   # pooled-AUPRC histogram resolution over |reg| in [0, 1]
ZMAX = 20.0     # --standardize tissue: |z| is unbounded; the pooled histogram bins |z|/ZMAX (clipped)

# compact indices (position in LABEL_LIST_SEGMENTATION) of the anatomy head's classes
TISSUE_GROUPS = {"WM": [20, 38], "cortex": [21, 39], "latvent": [22, 40], "inflat": [23, 41],
                 "3rd4th": [1, 2], "sulcalCSF": [4], "thal": [26, 44], "caud": [27, 45],
                 "put": [28, 46], "pall": [29, 47], "hipp": [30, 48], "amyg": [31, 49],
                 "cbWM": [24, 42], "cbCtx": [25, 43], "brainstem": [3]}
LESION_IDX = 7
MIN_CLASS_VOX = 2000


def _robust(v):
    med = float(np.median(v)); mad = float(np.median(np.abs(v - med))) * 1.4826
    return med, max(mad, 1e-6)


def standardize_scan(sreg, brain):
    """Signed map -> robust z over the scan's own brain mask: (s - median) / (1.4826 MAD). One scale per scan, no labels: the
    calibration a single dataset-wide threshold needs when the residual's scale drifts with the input frame (2026-09-30)."""
    v = sreg[brain]; med = float(np.median(v)); mad = float(np.median(np.abs(v - med))) * 1.4826 + 1e-6
    return (sreg - med) / mad


def standardize_tissue(sreg, brain, seg, per_class_scale=True):
    """Signed map -> per-tissue robust z inside the brain, using the derivative's OWN
    anatomy labels (no disease prior). Each class with >= MIN_CLASS_VOX voxels is
    standardized against itself; smaller classes and unassigned labels against the
    pooled healthy (non-lesion) brain. Lesion-labeled voxels are NOT standardized
    against themselves (that would erase a detected lesion): they use the WM statistics.
    Returns (z, fraction of brain voxels that used a fallback: lesion/WM, small class,
    or unassigned -> pooled). per_class_scale=False ("center"): subtract the class
    median but divide every class by the pooled healthy MAD, so a class-wise offset is
    removed without magnifying the tails of quiet classes."""
    z = np.zeros_like(sreg, dtype=np.float32)
    healthy = brain & (seg != LESION_IDX)
    pooled = _robust(sreg[healthy]) if healthy.sum() >= MIN_CLASS_VOX else _robust(sreg[brain])
    gmad = pooled[1]
    assigned = np.zeros_like(brain); fallback = 0
    wm_sel = brain & np.isin(seg, TISSUE_GROUPS["WM"])
    wm_stats = _robust(sreg[wm_sel]) if wm_sel.sum() >= MIN_CLASS_VOX else pooled
    for ids in TISSUE_GROUPS.values():
        sel = brain & np.isin(seg, ids); n = int(sel.sum())
        if n == 0:
            continue
        if n >= MIN_CLASS_VOX:
            med, mad = _robust(sreg[sel])
        else:
            med, mad = pooled; fallback += n
        z[sel] = (sreg[sel] - med) / (mad if per_class_scale else gmad); assigned |= sel
    les = brain & (seg == LESION_IDX)
    if les.any():
        z[les] = (sreg[les] - wm_stats[0]) / (wm_stats[1] if per_class_scale else gmad); assigned |= les; fallback += int(les.sum())
    rest = brain & ~assigned
    if rest.any():
        z[rest] = (sreg[rest] - pooled[0]) / pooled[1]; fallback += int(rest.sum())
    return z, fallback / max(int(brain.sum()), 1)


def apply_postproc(sreg, spec, zooms=None):
    """--postproc median:K -- 3-D median filter on the SIGNED map before every metric: the post-processing the UAD benchmark
    (2512.01534) applies to every method, on scans it had RESAMPLED TO 1 mm ISOTROPIC. Our maps sit on the native grid, so K is
    a length in millimetres and the kernel is K/zoom voxels per axis (min 1): K^3 voxels on 1 mm data (ATLAS, BraTS), (4, 4, 1) on
    the 3 mm-slice WMH grids. Before 2026-10-02 the kernel was K voxels per axis, i.e. 12 mm through-plane on WMH -- that is what
    cost WMH 4-8 AUROC points. medianvox:K keeps the voxel-kernel behaviour for reference. Fixed a priori, never tuned on a test leg."""
    if not spec:
        return sreg
    kind, _, k = spec.partition(":")
    from scipy import ndimage
    if kind == "median":
        if zooms is None:
            size = int(k)
        else:
            size = tuple(max(1, int(round(float(k) / float(z)))) for z in zooms[:3])
        return ndimage.median_filter(sreg, size=size, mode="nearest").astype(np.float32)
    if kind == "medianvox":
        return ndimage.median_filter(sreg, size=int(k), mode="nearest").astype(np.float32)
    raise ValueError(f"--postproc {spec!r}: expected median:K (mm) or medianvox:K (voxels)")


def pooled_hist(pos, neg):
    """Per-case score histograms (positives, negatives) to accumulate a whole-dataset
    precision-recall curve without holding every voxel: the UAD benchmark
    (arXiv 2512.01534) computes AUPRC "on the complete dataset once", i.e. over all
    voxels of all cases pooled, and treats it as the primary threshold-free metric."""
    hp = np.bincount(np.clip((pos * (NBINS - 1)).astype(np.int64), 0, NBINS - 1), minlength=NBINS)
    hn = np.bincount(np.clip((neg * (NBINS - 1)).astype(np.int64), 0, NBINS - 1), minlength=NBINS)
    return hp, hn


def ceil_dice_from_hist(hp, hn):
    """Per-case best Dice over all thresholds on |reg| (the benchmark's ceil-Dice, an
    ORACLE ceiling: the threshold uses the case's own ground truth). Reported only in
    the benchmark-comparison table, never as a result."""
    G = hp.sum()
    if G == 0:
        return float("nan")
    tp = np.cumsum(hp[::-1]); fp = np.cumsum(hn[::-1])
    return float((2.0 * tp / (tp + fp + G)).max())


def ceil_dice_global(hists):
    """ONE threshold for the whole dataset, chosen to maximise the mean per-case Dice over the cases
    scored (the benchmark tunes a single threshold on a lesion-containing split; this is that protocol's
    closest analogue, tuned on the scored cases themselves, so still an oracle: comparison table only).
    hists = [(hp, hn), ...] per case."""
    cases = [(hp, hn) for hp, hn in hists if hp.sum() > 0]
    if not cases:
        return float("nan")
    curves = []
    for hp, hn in cases:
        tp = np.cumsum(hp[::-1]); fp = np.cumsum(hn[::-1]); curves.append(2.0 * tp / (tp + fp + hp.sum()))
    return float(np.mean(curves, axis=0).max())


def ceil_dice_global_thr(hists):
    """ceil_dice_global() plus the histogram-bin threshold that attains it: predicted = bins >= b*."""
    cases = [(hp, hn) for hp, hn in hists if hp.sum() > 0]
    if not cases:
        return float("nan"), None
    curves = []
    for hp, hn in cases:
        tp = np.cumsum(hp[::-1]); fp = np.cumsum(hn[::-1]); curves.append(2.0 * tp / (tp + fp + hp.sum()))
    mean_curve = np.mean(curves, axis=0); k = int(mean_curve.argmax())
    return float(mean_curve[k]), NBINS - 1 - k


def hd95_mm(pred, gt, zooms):
    """95% Hausdorff distance in mm between two binary volumes: the 95th percentile of the pooled
    surface-to-surface distances in both directions (MedPy's hd95 convention). nan if either is empty."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    if not pred.any() or not gt.any():
        return float("nan")
    sp = pred & ~binary_erosion(pred); sg = gt & ~binary_erosion(gt)
    d_to_gt = distance_transform_edt(~sg, sampling=zooms)[sp]      # every pred-surface voxel to the nearest GT-SURFACE voxel (MedPy)
    d_to_pred = distance_transform_edt(~sp, sampling=zooms)[sg]    # every GT-surface voxel to the nearest pred-surface voxel
    return float(np.percentile(np.concatenate([d_to_gt, d_to_pred]), 95))


def auprc_from_hist(hp, hn):
    """Step-wise AP from pooled histograms, thresholds descending; negatives first
    within a bin (conservative), matching average_precision()."""
    if hp.sum() == 0:
        return float("nan")
    tp = np.cumsum(hp[::-1]); fp = np.cumsum(hn[::-1])
    prec = tp / np.maximum(tp + fp, 1); dtp = hp[::-1]
    return float((prec * dtp).sum() / hp.sum())


PF_FMT = ((8, 3), (8, 3), (6, 3), (8, 3), (10, 3), (6, 1), (9, 3), (9, 3), (7, 3), (7, 3))   # (width, decimals) per PF_COLS; n_fp median can be x.5
PF_HDR = " " + " ".join(f"{c:>{w}s}" for c, (w, _) in zip(PF_COLS, PF_FMT)) + f" {'DiceCeilG':>9s} {'DiceCeilP':>9s}"   # P = pooled voxels, one threshold (Lagogiannis 2023 compute_best_dice / DISYRE's own papers)

_CTX = {}   # set by main() before the pool forks; read by _score_case in every worker


def _score_case(idx_case):
    """One case, every deriv x mask: returns (printable lines, summary records, missing-seg count)."""
    idx, case = idx_case
    a, droot, data_dir, wanted, EX, std = _CTX["a"], _CTX["droot"], _CTX["data_dir"], _CTX["wanted"], _CTX["EX"], _CTX["std"]
    rng = np.random.default_rng([0, idx])   # per-case stream: identical in serial and parallel runs
    out, recs, missing = [], [], 0
    sub, t1, gtp = case[0], case[1], case[2]
    if a.gt_dir:
        gtp = os.path.join(droot, a.gt_dir, output_name_for(t1, data_dir, a.dataset, a.modality).replace("_seg.nii.gz", "_gt.nii.gz"))
    if not os.path.isfile(gtp):
        out.append(f"{sub:16s} (no ground truth at {gtp})"); return out, recs, missing
    g_n = nib.load(gtp)
    gt = np.isin(g_n.get_fdata().astype(np.int32), wanted)
    vox = float(np.prod(g_n.header.get_zooms()[:3]))
    if gt.sum() == 0:   # only an EMPTY ground truth is unscorable; no lesion-size floor (none in the benchmark, DISYRE or Lagogiannis; 2026-09-30)
        out.append(f"{sub:16s} (ground truth empty)"); return out, recs, missing
    name = output_name_for(t1, data_dir, a.dataset, a.modality)
    # mask name: T1w-named when a modality swap happened
    mask_name = output_name_for(case[3], data_dir, a.dataset, "3DT1") if len(case) > 3 else name
    for deriv in a.derivs:
        rp = os.path.join(droot, deriv, name.replace("_seg.nii.gz", "_reg.nii.gz"))
        if not os.path.isfile(rp):
            out.append(f"{sub:16s} {deriv[:34]:34s} (no reg map)"); continue
        _rn = nib.load(rp); sreg = apply_postproc(_rn.get_fdata().astype(np.float32) * 2.0 - 1.0, a.postproc, _rn.header.get_zooms())
        reg = np.abs(sreg)
        seg = None
        if std and a.standardize != "scan":
            sp = os.path.join(droot, deriv, name)
            if os.path.isfile(sp):
                seg = np.asarray(nib.load(sp).dataobj).astype(np.int32)
            else:
                missing += 1
                out.append(f"# WARNING: no anatomy labels at {sp}; such cases score |reg| unstandardized")
        for mask in a.masks:
            if mask == "img>0":
                brain = nib.load(t1).get_fdata() > 0
            else:
                mp = os.path.join(droot, mask, mask_name)
                if not os.path.isfile(mp):
                    out.append(f"{sub:16s} {deriv[:34]:34s} {mask[:30]:30s} (no mask)"); continue
                m = nib.load(mp).get_fdata().astype(np.int32)
                brain = (m > 0) & ~np.isin(m, EX)
            if brain.shape != gt.shape or brain.sum() < 1000:
                out.append(f"{sub:16s} {deriv[:34]:34s} {mask[:30]:30s} (shape/size)"); continue
            G = gt.sum()
            d = {}
            fb = float("nan")
            use_std = (a.standardize == "scan") or (seg is not None and seg.shape == sreg.shape)
            if a.standardize == "scan":
                s_use = standardize_scan(sreg, brain); r_use = np.abs(s_use)
            elif use_std:
                s_use, fb = standardize_tissue(sreg, brain, seg, per_class_scale=(a.standardize == "tissue")); r_use = np.abs(s_use)
            else:
                s_use, r_use = sreg, reg
            if a.rule == "z" and use_std:
                thr = a.z   # already in robust-sigma units per tissue; no second pass
                pols = (("bright", s_use > thr), ("dark", s_use < -thr), ("abs", r_use > thr))
            elif a.rule == "z":
                aa = reg[brain]
                med = np.median(aa); mad = np.median(np.abs(aa - med)) * 1.4826 + 1e-6
                thr = med + a.z * mad
                pols = (("bright", sreg > thr), ("dark", sreg < -thr), ("abs", reg > thr))
            elif a.rule == "cluster":
                # blind spatial rule: z-forming cut on the signed map's own null, then a
                # minimum connected-component extent (eval/blind_threshold.py)
                mu = np.median(s_use[brain]); dev = s_use - mu
                sig = 1.4826 * (np.median(np.abs(s_use[brain] - mu)) + 1e-9)
                thr = a.z
                pols = tuple(zip(("bright", "dark", "abs"), cluster_mask(dev, brain, sig, zc=a.z, min_vox=a.min_vox)))
            else:
                # per-image FDR cut on the SIGNED map's own null (eval/blind_threshold.py);
                # the thr column then reports the cut in robust-sigma units
                mu, cb, cd = threshold_sides(s_use[brain], q=a.q, null=a.null)
                dev = s_use - mu
                sig = 1.4826 * (np.median(np.abs(s_use[brain] - mu)) + 1e-9)
                thr = min(cb, cd) / sig
                pols = (("bright", dev >= cb), ("dark", dev <= -cd), ("abs", (dev >= cb) | (dev <= -cd)))
            for pol, pos in pols:
                p = pos & brain
                d[pol] = 2 * (p & gt).sum() / max(p.sum() + G, 1)
            sa, sb = auc_subsample(r_use[brain & gt], r_use[brain & ~gt], rng)
            A = auc_from_subsample(sa, sb)
            P = average_precision(r_use[brain & gt], r_use[brain & ~gt])
            r_hist = np.minimum(r_use / ZMAX, 1.0) if use_std else r_use   # histogram domain [0, 1]; same decision as r_use
            row=(brain.sum()*vox/1000.0, thr, A, P, d['bright'], d['dark'], d['abs'],
                 (gt & ~brain).sum()/G)
            pf = ""
            if a.prevfree:
                pfv = prevfree_metrics(sa, sb, r_use, brain, gt, dict(pols)["abs"])
                row = row + pfv
                pf = " " + " ".join(f"{v:{w}.{p}f}" for v, (w, p) in zip(pfv, PF_FMT))
            if a.summary:
                hp, hn = pooled_hist(r_hist[brain & gt], r_hist[brain & ~gt])
                recs.append((deriv, mask, row, hp, hn, ceil_dice_from_hist(hp, hn)))
            else:
                out.append(f"{sub:16s} {deriv[:34]:34s} {mask[:30]:30s} "
                      f"{row[0]:8.0f} {thr:7.4f} {A:6.3f} {P:6.3f} "
                      f"{d['bright']:8.3f} {d['dark']:7.3f} {d['abs']:7.3f} {row[7]:7.4f}" + (f" {fb:7.4f}" if std else "") + pf)
    return out, recs, missing


def _hd95_case(idx_case):
    """--hd95 second pass: HD95 (mm) of the binarization at the single global oracle threshold
    (DiceCeilG's bin b*, _CTX["bstar"][(deriv, mask)]) against the brain-restricted GT, one case,
    every deriv x mask. Same loading and standardization as _score_case. Returns [(deriv, mask, hd95,
    pred_empty, gt_empty)]."""
    idx, case = idx_case
    a, droot, data_dir, wanted, EX, std, bstar = (_CTX["a"], _CTX["droot"], _CTX["data_dir"], _CTX["wanted"],
                                                 _CTX["EX"], _CTX["std"], _CTX["bstar"])
    res = []
    sub, t1, gtp = case[0], case[1], case[2]
    if a.gt_dir:
        gtp = os.path.join(droot, a.gt_dir, output_name_for(t1, data_dir, a.dataset, a.modality).replace("_seg.nii.gz", "_gt.nii.gz"))
    if not os.path.isfile(gtp):
        return res
    g_n = nib.load(gtp)
    gt = np.isin(g_n.get_fdata().astype(np.int32), wanted)
    zooms = tuple(float(z) for z in g_n.header.get_zooms()[:3])
    if gt.sum() == 0:
        return res
    name = output_name_for(t1, data_dir, a.dataset, a.modality)
    mask_name = output_name_for(case[3], data_dir, a.dataset, "3DT1") if len(case) > 3 else name
    for deriv in a.derivs:
        rp = os.path.join(droot, deriv, name.replace("_seg.nii.gz", "_reg.nii.gz"))
        if not os.path.isfile(rp):
            continue
        _rn = nib.load(rp); sreg = apply_postproc(_rn.get_fdata().astype(np.float32) * 2.0 - 1.0, a.postproc, _rn.header.get_zooms())
        reg = np.abs(sreg)
        seg = None
        if std and a.standardize != "scan":
            sp = os.path.join(droot, deriv, name)
            if os.path.isfile(sp):
                seg = np.asarray(nib.load(sp).dataobj).astype(np.int32)
        for mask in a.masks:
            b = bstar.get((deriv, mask))
            if b is None:
                continue
            if mask == "img>0":
                brain = nib.load(t1).get_fdata() > 0
            else:
                mp = os.path.join(droot, mask, mask_name)
                if not os.path.isfile(mp):
                    continue
                m = nib.load(mp).get_fdata().astype(np.int32)
                brain = (m > 0) & ~np.isin(m, EX)
            if brain.shape != gt.shape or brain.sum() < 1000:
                continue
            use_std = (a.standardize == "scan") or (seg is not None and seg.shape == sreg.shape)
            if a.standardize == "scan":
                s_use = standardize_scan(sreg, brain); r_use = np.abs(s_use)
            elif use_std:
                s_use, _ = standardize_tissue(sreg, brain, seg, per_class_scale=(a.standardize == "tissue")); r_use = np.abs(s_use)
            else:
                r_use = reg
            r_hist = np.minimum(r_use / ZMAX, 1.0) if use_std else r_use
            pred = (np.clip((r_hist * (NBINS - 1)).astype(np.int64), 0, NBINS - 1) >= b) & brain   # exactly the histogram's cut
            gt_b = gt & brain
            res.append((deriv, mask, hd95_mm(pred, gt_b, zooms), not pred.any(), not gt_b.any()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--derivs", nargs="+", required=True)
    ap.add_argument("--masks", nargs="+", required=True)
    ap.add_argument("--modality", default="3DT1")
    ap.add_argument("--list-modality", default=None)
    ap.add_argument("--image-name", default=None)
    ap.add_argument("--target", type=int, nargs="+", default=None)
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--rule", default="z", choices=["z", "fdr", "cluster"], help="z: median+z*MAD of |reg| (default); fdr: per-image Benjamini-Hochberg on the signed map's null; cluster: z-forming cut (--z) on the signed null + minimum extent (--min_vox)")
    ap.add_argument("--q", type=float, default=0.05, help="FDR level for --rule fdr")
    ap.add_argument("--min_vox", type=int, default=27, help="minimum component size for --rule cluster")
    ap.add_argument("--null", default="gaussian", choices=["gaussian", "laplace", "twopiece"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gt_dir", default=None,
                    help="Override the ground truth: <derivatives>/<gt_dir>/<derivative name with _seg -> _gt>.nii.gz "
                         "(e.g. gt_3DT1 for WMH scored in 3D-T1 space with the FLAIR-drawn GT rigidly resampled).")
    ap.add_argument("--workers", type=int, default=1, help="cases scored in parallel (fork); output identical to --workers 1")
    ap.add_argument("--summary", action="store_true",
                    help="one median row per (reg map, mask) instead of per case")
    ap.add_argument("--postproc", default="", help="median:K -- 3-D median filter on the signed map before every metric (DISYRE-style post-processing; kernel fixed a priori)")
    ap.add_argument("--standardize", default="none", choices=["none", "tissue", "center", "scan"],
                    help="scan: |z| of the signed map standardized ONCE per scan (median/MAD over the brain mask; no labels, GT-free; 2026-09-30). "
                         "tissue: score |z| of the signed map standardized per anatomy-head class inside the brain "
                         "(the --z cut then applies to |z| directly); none: score |reg| (default, unchanged)")
    ap.add_argument("--prevfree", action="store_true",
                    help="append prevalence-free columns (see prevfree_metrics): TPR at 1%%/5%% voxel FPR, "
                         "partial AUC over FPR<=.05 normalized as pAUC/.05 (not McClish), lesion-wise "
                         "sensitivity (>=1 voxel / >=10%% of a >=20-vox 26-conn GT component above the fixed "
                         "cut used for D_abs), false-positive components (>=20 vox, no GT) per case, and "
                         "max score in GT / healthy 99.9th percentile. Medians over cases in --summary.")
    ap.add_argument("--hd95", action="store_true",
                    help="with --summary --prevfree: a second pass reports the 95%% Hausdorff distance (mm, MedPy pooled "
                         "convention) of the binarization at the single global oracle threshold (DiceCeilG) against the "
                         "brain-restricted GT: median [IQR] over cases where both sets are non-empty, plus the empty counts")
    a = ap.parse_args()
    if a.hd95 and not (a.summary and a.prevfree):
        sys.exit("# --hd95 needs --summary --prevfree (the global oracle threshold)")

    data_dir = os.environ.get('PR_EVAL_DATA_ROOT', '/path/to/eval_data_root') + f"/{a.dataset}"
    droot = os.environ.get('PR_DERIV_ROOT', '/path/to/deriv_root') + f"/{a.dataset}/derivatives"
    key = dataset_key(a.dataset)
    wanted = a.target if a.target else ([1, 2, 3] if key == "BraTS" else [1])

    cases = list_cases(data_dir, a.dataset, modality=a.list_modality)
    # BIDS suffix swap (BraTS/ATLAS on FLAIR etc.), same as auc_brain.py:112.
    # The 4th element keeps the T1w path, because the MASK is T1w-named even
    # for a FLAIR run -- one mask serves both modalities.
    # BraTS/ATLAS: output_name_for() reads the modality from the image PATH, not from
    # --modality, so a contrast swap needs the swapped path. Without --image-name the
    # 2026-09-18 BraTS FLAIR legs scored the T1w maps against the whole tumour (AUC 0.37-0.52).
    if (not a.image_name and not a.dataset.startswith("WMH") and a.modality not in (None, "3DT1", "T1", "T1w")):
        a.image_name = f"_{a.modality}.nii.gz"
        print(f"# --modality {a.modality} on {a.dataset}: using --image-name {a.image_name} (path carries the modality)")
    if a.image_name and a.image_name.startswith("_"):
        cases = [(s, t.replace("_T1w.nii.gz", a.image_name), g, t) for s, t, g in cases]
        cases = [c for c in cases if os.path.isfile(c[1])]
    elif a.image_name:
        cases = [(s, os.path.join(os.path.dirname(t), a.image_name), g) for s, t, g in cases]
        cases = [c for c in cases if os.path.isfile(c[1])]
    if a.subjects:
        cases = [c for c in cases if c[0] in a.subjects]
    if a.limit:
        cases = cases[:a.limit]
    acc = {}
    pooled = {}    # (deriv, mask) -> [hist_pos, hist_neg] over all cases, for whole-dataset AUPRC
    ceil = {}      # (deriv, mask) -> per-case oracle max-Dice (benchmark's ceil-Dice; comparison only)
    hists = {}     # (deriv, mask) -> per-case (hp, hn), for the single-global-threshold oracle (--prevfree column DiceCeilG)
    std = a.standardize != "none"
    missing_seg = 0
    print(f"# {a.dataset}  n_cases={len(cases)}  z={a.z}  labels={wanted}" + (f"  standardize={a.standardize}" if std else "") + (f"  postproc={a.postproc}{'mm' if a.postproc.startswith('median:') else ''}" if a.postproc else ""))
    if not a.summary: print(f"{'subject':16s} {'reg map':34s} {'mask':30s} "
          f"{'mask_mL':>8s} {'thr':>7s} {'AUC':>6s} {'AP':>6s} "
          f"{'D_bright':>8s} {'D_dark':>7s} {'D_abs':>7s} {'gt_out':>7s}" + (f" {'fb_frac':>7s}" if std else "") + (PF_HDR if a.prevfree else ""))

    _CTX.update(a=a, droot=droot, data_dir=data_dir, wanted=wanted, EX=EX, std=std)
    warned = False
    if a.workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(a.workers)
        results = pool.imap(_score_case, list(enumerate(cases)), chunksize=1)
    else:
        results = map(_score_case, enumerate(cases))
    for out, recs, missing in results:
        missing_seg += missing
        for line in out:
            if line.startswith("# WARNING"):
                if warned: continue
                warned = True
            print(line)
        for deriv, mask, row, hp, hn, cd in recs:
            acc.setdefault((deriv, mask), []).append(row)
            ceil.setdefault((deriv, mask), []).append(cd)
            hists.setdefault((deriv, mask), []).append((hp, hn))
            if (deriv, mask) in pooled:
                pooled[(deriv, mask)][0] += hp; pooled[(deriv, mask)][1] += hn
            else:
                pooled[(deriv, mask)] = [hp, hn]
    if a.workers > 1:
        pool.close(); pool.join()

    if std:
        print(f"# standardize={a.standardize}: {missing_seg} (deriv, case) pairs had no anatomy labels and scored |reg| unstandardized")
    if a.summary:
        import statistics as _st
        # sorted by AUC, the primary metric (2026-09-17); AP is threshold-free too but prevalence-aware
        # AUC/AP: medians over cases. AUPRCpool: one PR curve over all voxels of all cases
        # (the benchmark's convention). Sorted by AUC, the primary metric.
        # DiceCeilG = one global threshold, per-case Dice averaged (benchmark 2512.01534 protocol); comparison table only.
        print(f"{'reg map':78s} {'mask':16s} {'n':>4s} {'vol_mL':>7s} {'AUC':>6s} {'AP':>6s} {'AUPRCpool':>9s} {'DiceCeilG':>9s} "
              f"{'D_bright':>9s} {'D_dark':>7s} {'D_abs':>7s} {'gt_out':>7s}" + (PF_HDR if a.prevfree else ""))
        for (deriv, mask), rs in sorted(acc.items(), key=lambda kv: -_st.median([r[2] for r in kv[1]])):
            m = lambda i: _st.median([r[i] for r in rs])
            # DiceCeilG: ONE global threshold maximising the mean per-case Dice = the benchmark's ceil-Dice protocol;
            # the per-case oracle (mean of per-case best thresholds) is never reported -- it runs ~1.5x higher.
            pa = auprc_from_hist(*pooled[(deriv, mask)]); dc = ceil_dice_global(hists[(deriv, mask)])
            print(f"{deriv[:78]:78s} {mask[:16]:16s} {len(rs):4d} {m(0):7.0f} {m(2):6.3f} {m(3):6.3f} {pa:9.3f} {dc:9.3f} "
                  f"{m(4):9.3f} {m(5):7.3f} {m(6):7.3f} {m(7):7.4f}"
                  + ("".join(f" {np.nanmedian([r[8 + i] for r in rs]):{w}.{p}f}" for i, (w, p) in enumerate(PF_FMT)) + f" {ceil_dice_global(hists[(deriv, mask)]):9.3f} {ceil_dice_from_hist(*pooled[(deriv, mask)]):9.3f}" if a.prevfree else ""))

    if a.hd95:
        if not (a.summary and a.prevfree):
            print("# --hd95 needs --summary --prevfree (the global oracle threshold)", file=sys.stderr); return
        bstar = {k: ceil_dice_global_thr(h)[1] for k, h in hists.items()}
        _CTX["bstar"] = {k: b for k, b in bstar.items() if b is not None}
        hd = {}
        if a.workers > 1:
            import multiprocessing as mp
            pool2 = mp.get_context("fork").Pool(a.workers)
            res2 = pool2.imap(_hd95_case, list(enumerate(cases)), chunksize=1)
        else:
            res2 = map(_hd95_case, enumerate(cases))
        for rows in res2:
            for deriv, mask, h, pe, ge in rows:
                hd.setdefault((deriv, mask), []).append((h, pe, ge))
        if a.workers > 1:
            pool2.close(); pool2.join()
        print(f"# HD95 (mm) at the single global oracle threshold (DiceCeilG cut, brain-restricted GT; MedPy pooled 95th pct)")
        print(f"{'reg map':78s} {'mask':16s} {'n':>4s} {'HD95med':>8s} {'IQR25':>7s} {'IQR75':>7s} {'HD95mean':>8s} {'predEmpty':>9s} {'gtEmpty':>7s} {'bin*':>6s} {'cut':>7s}")
        for (deriv, mask), rows in sorted(hd.items()):
            v = np.array([h for h, pe, ge in rows if np.isfinite(h)])
            pe = sum(1 for _, pe, _ in rows if pe); ge = sum(1 for _, _, ge in rows if ge)
            if v.size:
                print(f"{deriv[:78]:78s} {mask[:16]:16s} {v.size:4d} {np.median(v):8.2f} {np.percentile(v, 25):7.2f} {np.percentile(v, 75):7.2f} {v.mean():8.2f} {pe:9d} {ge:7d} {bstar[(deriv, mask)]:6d} {bstar[(deriv, mask)] / (NBINS - 1) * (ZMAX if std else 1.0):7.3f}")
            else:
                print(f"{deriv[:78]:78s} {mask[:16]:16s} {0:4d} {'nan':>8s} {'':7s} {'':7s} {'':8s} {pe:9d} {ge:7d}")


if __name__ == "__main__":
    main()
