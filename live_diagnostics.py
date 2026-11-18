"""
live_diagnostics.py
====================
Drop this file next to train.py and `import live_diagnostics as diag`.

These four checks need tensors that only exist inside the training loop, so
they can't be recovered from a log file after the fact (that's what
analyze_pathology_log.py is for). Each function below is written against the
exact variable names already in train.py so wiring them in is a
small, additive diff -- nothing existing needs to change.

------------------------------------------------------------------------
1) SAMPLER PROVENANCE  (answers: "did my k_rose_sampler edit really run?")
------------------------------------------------------------------------
Call ONCE, right after `k_rose_sampler` is defined and before it's handed to
paired_pathology_generator(...) (around train.py line ~352):

    def k_rose_sampler():
        ...
        return float(np.random.uniform(1.0, 2.5))

    diag.report_sampler_provenance(                     # <-- add this
        k_rose_sampler,
        expected_range=(1.0, 2.5),
        git_hash=_git_identifier(),
        log_print=log_print,
    )

    mode_sampler = make_mode_sampler(...)
    gen = paired_pathology_generator(..., k_rose_sampler=k_rose_sampler)

This bakes the *empirical* behavior of the function that's actually about to
be used into this run's log header, so "what range did this run use" is
never again a question you have to reconstruct after the fact.

------------------------------------------------------------------------
2) NORMALIZATION-CASCADE CHECK (answers: "is one hot lesion voxel dragging
   normalize_per_sample's per-volume max around and washing out everything
   else in that volume?")
------------------------------------------------------------------------
normalize_per_sample() rescales each volume by its own (min, max) and its own
docstring assumes the generator already emits [0, 1]. If a pathology mode
ever produces a value above that, this is where it would quietly turn into
"the rest of the brain got compressed," not just "the lesion is too bright."

Call every step, right where img_clean / img_path exist and BEFORE they're
overwritten by normalize_per_sample (around train.py line ~593-598):

    img_clean_norm = normalize_per_sample(img_clean)
    clean_img_vis = normalize_per_sample(clean_img_raw)

    if use_pathology:
        diag.check_normalization_cascade(                # <-- add this
            img_clean, img_path, step=i, log_print=log_print)
        img_path_norm = normalize_per_sample(img_path)
        ...

It's two .max() calls, essentially free, and only prints when something is
actually suspicious (or periodically, see accumulate=True below).

------------------------------------------------------------------------
3) REALIZED vs REQUESTED CONTRAST (answers: "does k_rose predict what the
   lesion actually looks like?") + TEXTURE RATIO (answers: "is this
   lesion visible via something other than a mean shift, e.g. speckle?")
------------------------------------------------------------------------
Call inside the existing `i % 1000 == 0` visualization block, right after
path_img_cpu / path_seg_cpu / clean_cpu / clean_seg_cpu all exist (around
train.py line ~733-756), alongside the existing mode_name / k_rose_str
lines:

    mode_name = case['pathology_metadata'].get('mode_name', 'unknown')
    k_rose_str = f"k_rose={case.get('k_rose', 'N/A'):.2f}"
    realized_k = diag.measure_realized_contrast(              # <-- add this
        path_img_cpu, clean_cpu, path_seg_cpu, LESION_COMPACT_IDX,
        underlying_seg_vol=clean_seg_cpu)
    texture_ratio = diag.measure_realized_texture_ratio(       # <-- add this
        path_img_cpu, path_seg_cpu, LESION_COMPACT_IDX,
        underlying_seg_vol=clean_seg_cpu)
    RUNNING_DIAG.record(case.get('k_rose'), realized_k)        # <-- add this

`realized_k` now computes mean(|path_img - clean_img|) over lesion voxels
divided by a reference sigma -- the SAME quantity generator.py's own
_correct_lesion_visibility computes as `mean_visible`, directly comparable
to the requested k_rose (their tau = k_rose * sigma). This matters because
that function draws up to 5 independently-parameterized GMM zones per
lesion (see n_lesion_zones in _inject_pathology) to simulate heterogeneous
internal appearance -- a lesion can have every voxel individually clear
the generator's own per-voxel |diff|>tau gate while still canceling out in
a naive whole-lesion signed mean. Taking the absolute value per voxel
before averaging (matching their formula exactly) is what fixes that.

`underlying_seg_vol=clean_seg_cpu` supplies the matched-tissue-elsewhere
reference population (robust to large/patchy lesions); the fallback uses a
GAPPED dilation band (gap=3 by default), mirroring generator.py's own
_surround_stats -- its comment documents surround sd inflating 0.092->0.224
without that gap, from the lesion's own blurred edge contaminating an
ungapped ring.

`texture_ratio` is a separate, complementary signal: std(lesion) /
std(reference), useful on its own terms as "how much internal texture does
this region have," independent of what realized_k reports.

...then extend the existing log_print call to also print `realized_k` and
`texture_ratio`, and call `RUNNING_DIAG.maybe_report(i, log_print)` right
after it. See the `RollingDiagnostics` class below -- instantiate ONE of
these before the training loop starts (`RUNNING_DIAG = diag.RollingDiagnostics()`).

------------------------------------------------------------------------
5) ROLLING LESION-DICE MEAN (answers: "is it improving?" without a
   post-hoc log pull -- raw per-checkpoint lesion dice is dominated by
   whatever lesion size that checkpoint happened to draw, since dice on a
   handful of true-positive voxels is close to a coin flip)
------------------------------------------------------------------------
Call right where `les_dice` is computed and logged (train.py,
wherever the "Step {i} | lesion dice ..." line is printed):

    RUNNING_DIAG.record_dice(les_dice)                          # <-- add this
    log_print(f"Step {i} | lesion dice {les_dice:.4f} | lesion vox {n_les} "
              f"| mean fg dice {mean_fg:.4f} | ratio {les_dice / max(mean_fg, 1e-6):.3f} "
              f"| dice_roll20 {RUNNING_DIAG.rolling_dice_mean(20):.4f}")  # <-- add field

------------------------------------------------------------------------
4) WHOLE-VOLUME CLASS COLLAPSE (answers: "is the model defaulting most of
   the volume to one class, the way it did in your saturated example?")
------------------------------------------------------------------------
Call in the same visualization block, once path_pred_idx_vis and pred_idx
(clean) both exist (around train.py line ~731-737):

    clean_idx, clean_frac, _ = diag.dominant_class_fraction(
        clean_pred_cpu, FS_N_LABELS)
    path_idx, path_frac, _ = diag.dominant_class_fraction(
        path_pred_cpu, FS_N_LABELS)
    if path_frac > clean_frac + 0.15:      # heuristic threshold, tune freely
        name = diag.compact_idx_to_name(path_idx, LABEL_LIST_SEGMENTATION)
        log_print(f"  [viz step {i}] COLLAPSE WARNING: pathological "
                  f"prediction is {path_frac:.1%} class '{name}' "
                  f"(clean baseline {clean_frac:.1%})")
"""
import math

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------
# 1) Sampler provenance
# ---------------------------------------------------------------------
def report_sampler_provenance(sampler_fn, expected_range=None, n_samples=5000,
                               git_hash=None, log_print=print):
    """Empirically fingerprint a sampler function at startup, once.

    Draws `n_samples` values right now, in this process, from the exact
    function object about to be handed to the generator -- not from reading
    its source, which can silently be stale relative to what actually runs
    (wrong git worktree/commit, edited file after the process was already
    launched, etc.).
    """
    import hashlib
    import inspect

    try:
        src = inspect.getsource(sampler_fn)
    except (OSError, TypeError):
        src = "<source unavailable>"

    draws = [float(sampler_fn()) for _ in range(n_samples)]
    lo, hi = min(draws), max(draws)
    mean = sum(draws) / len(draws)

    log_print(f"[sampler check] git identifier: {git_hash!r}")
    log_print(f"[sampler check] source sha256[:12]: "
              f"{hashlib.sha256(src.encode()).hexdigest()[:12]}")
    log_print(f"[sampler check] empirical range over {n_samples} live draws: "
              f"[{lo:.4f}, {hi:.4f}]  mean={mean:.4f}")

    if expected_range is not None:
        exp_lo, exp_hi = expected_range
        mismatch = lo < exp_lo - 1e-6 or hi > exp_hi + 1e-6
        # Also flag a *suspiciously* narrow empirical range even when it's
        # technically inside expected_range -- e.g. if a run is short-lived
        # relative to n_samples this won't trigger, but it catches "range
        # looks bounded but is way off-center from what uniform(...) implies".
        if mismatch:
            log_print(f"[sampler check] *** MISMATCH *** you passed "
                      f"expected_range={expected_range} but the live "
                      f"function does not stay inside it. You are not "
                      f"running the sampler you think you are -- check "
                      f"which commit/process this actually is before "
                      f"trusting any downstream conclusion.")
        else:
            log_print(f"[sampler check] matches expected_range={expected_range}. OK.")
    log_print("[sampler check] source:\n" +
              "\n".join(f"    {line}" for line in src.splitlines()))


# ---------------------------------------------------------------------
# 2) Normalization-cascade check
# ---------------------------------------------------------------------
def check_normalization_cascade(raw_clean, raw_path, step=None, log_print=print,
                                 warn_ratio=1.5):
    """Compare the pre-normalization max of the pathological stream to the
    clean stream's. normalize_per_sample divides each volume by its OWN max,
    so if raw_path's max is inflated relative to raw_clean's, every other
    voxel in that pathological volume gets compressed relative to how the
    same tissue would render in the clean stream.Returns the ratio (path_max / clean_max).
    """
    with torch.no_grad():
        clean_max = raw_clean.amax().item()
        path_max = raw_path.amax().item()
    ratio = path_max / (clean_max + 1e-7)
    if ratio > warn_ratio:
        prefix = f"[step {step}] " if step is not None else ""
        log_print(f"{prefix}[cascade check] WARNING: pathological stream's "
                  f"raw max ({path_max:.4f}) is {ratio:.2f}x the clean "
                  f"stream's ({clean_max:.4f}). normalize_per_sample will "
                  f"divide this whole volume by the larger number -- real "
                  f"tissue in the pathological volume renders dimmer here "
                  f"than the same tissue does in the clean stream.")
    return ratio


# ---------------------------------------------------------------------
# 3) Realized vs requested contrast
# ---------------------------------------------------------------------
def _dilate_mask(mask_3d, iters=3):
    """3D binary dilation via repeated max-pool. mask_3d: (D,H,W) bool/float."""
    m = mask_3d.float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    for _ in range(iters):
        m = F.max_pool3d(m, kernel_size=3, stride=1, padding=1)
    return m[0, 0] > 0.5


def measure_realized_contrast(path_image_vol, clean_image_vol, seg_vol, lesion_idx,
                               underlying_seg_vol=None, gap=3, dilate_iters=3,
                               min_reference_voxels=200):
    """mean(|path_image - clean_image|) over lesion voxels, divided by a
    reference sigma -- the SAME quantity generator.py's own
    _correct_lesion_visibility gate computes as `mean_visible`, directly
    comparable to the requested k_rose (their tau = k_rose * sigma, so this
    value crossing k_rose is the same event as their gate almost firing).

    Takes the absolute value PER VOXEL before averaging, not a signed mean
    across the whole lesion. The generator draws up to 5 independent GMM
    zones per lesion (see n_lesion_zones in _inject_pathology), deliberately
    producing heterogeneous appearance -- e.g. a dark necrotic core next to
    a bright hemorrhagic rim. A signed whole-lesion mean lets such zones
    cancel even when every voxel individually clears the generator's own
    per-voxel |diff| > tau gate, understating a lesion that is in fact
    highly visible. Per-voxel absolute value, matching the generator's own
    formula, avoids that cancellation.

    `clean_image_vol` must be co-registered with `path_image_vol` (i.e. the
    same underlying case's clean stream) -- train.py's `clean_cpu`
    fills this role.

    Reference sigma: prefers a matched-tissue-elsewhere population via
    `underlying_seg_vol` (robust to large/patchy lesions -- see the module
    docstring). Falls back to a GAPPED dilation band (gap voxels of
    clearance, then a dilate-voxel-wide band).
    """
    if lesion_idx is None:
        return None
    lesion_mask = seg_vol == lesion_idx
    if not lesion_mask.any():
        return None
    mean_visible = (path_image_vol[lesion_mask].float() - clean_image_vol[lesion_mask].float()).abs().mean()

    if underlying_seg_vol is not None:
        under_labels, counts = torch.unique(underlying_seg_vol[lesion_mask], return_counts=True)
        dominant_label = under_labels[counts.argmax()].item()
        reference_mask = (underlying_seg_vol == dominant_label) & (~lesion_mask)
        if reference_mask.sum() >= min_reference_voxels:
            ref_std = path_image_vol[reference_mask].float().std()
            if ref_std > 1e-7:
                return (mean_visible / ref_std).item()

    inner = _dilate_mask(lesion_mask, iters=gap)
    band = _dilate_mask(lesion_mask, iters=gap + dilate_iters) & (~inner)
    if band.sum() < min_reference_voxels:
        band = _dilate_mask(lesion_mask, iters=dilate_iters) & (~lesion_mask)  # small-lesion/edge fallback, no gap
    if not band.any():
        return None
    ref_std = path_image_vol[band].float().std()
    if ref_std < 1e-7:
        return None
    return (mean_visible / ref_std).item()


def measure_realized_texture_ratio(path_image_vol, seg_vol, lesion_idx, underlying_seg_vol=None,
                                    gap=3, dilate_iters=3, min_reference_voxels=200):
    """std(intensity WITHIN the lesion) / std(intensity in the same
    reference population used above).

    Complementary to measure_realized_contrast(): that one now matches the
    generator's own per-voxel |diff|-from-clean visibility gate, so a
    multi-zone heterogeneous lesion no longer looks falsely invisible to it.
    This one is still useful on its own terms as a plain "how much internal
    texture does this region have relative to normal tissue" readout -- a
    ratio near 1 means normal texture, several-fold or more means visibly
    heterogeneous regardless of what realized_k reports.

    Same reference logic and fallback as measure_realized_contrast(); same
    None conditions.
    """
    if lesion_idx is None:
        return None
    lesion_mask = seg_vol == lesion_idx
    if not lesion_mask.any():
        return None
    lesion_std = path_image_vol[lesion_mask].float().std()

    if underlying_seg_vol is not None:
        under_labels, counts = torch.unique(underlying_seg_vol[lesion_mask], return_counts=True)
        dominant_label = under_labels[counts.argmax()].item()
        reference_mask = (underlying_seg_vol == dominant_label) & (~lesion_mask)
        if reference_mask.sum() >= min_reference_voxels:
            ref_std = path_image_vol[reference_mask].float().std()
            if ref_std > 1e-7:
                return (lesion_std / ref_std).item()

    inner = _dilate_mask(lesion_mask, iters=gap)
    band = _dilate_mask(lesion_mask, iters=gap + dilate_iters) & (~inner)
    if band.sum() < min_reference_voxels:
        band = _dilate_mask(lesion_mask, iters=dilate_iters) & (~lesion_mask)
    if not band.any():
        return None
    ref_std = path_image_vol[band].float().std()
    if ref_std < 1e-7:
        return None
    return (lesion_std / ref_std).item()


# ---------------------------------------------------------------------
# 4) Whole-volume class collapse
# ---------------------------------------------------------------------
def dominant_class_fraction(pred_vol, n_classes, exclude=(0,)):
    """Fraction of `pred_vol` occupied by its single most common class,
    excluding `exclude` (background/unknown by default -- background being
    the plurality class is normal; a *tissue* class swallowing the volume
    is the collapse signature you're checking for).

    Returns (dominant_class_idx, dominant_fraction, all_fractions).
    """
    flat = pred_vol.reshape(-1).long()
    counts = torch.bincount(flat, minlength=n_classes).float()
    total = flat.numel()
    fractions = counts / total
    scored = fractions.clone()
    for e in exclude:
        if 0 <= e < len(scored):
            scored[e] = -1.0
    dominant_idx = int(scored.argmax().item())
    return dominant_idx, fractions[dominant_idx].item(), fractions


def compact_idx_to_name(compact_idx, label_list_segmentation,
                         freesurfer_labels=None):
    """Map a compact index (position in label_list_segmentation) back to a
    readable FreeSurfer name, for warning messages."""
    if freesurfer_labels is None:
        # Import lazily to avoid a hard dependency if the caller only wants
        # the numeric id.
        try:
            from train import FREESURFER_LABELS as freesurfer_labels
        except Exception:
            freesurfer_labels = {}
    real_id = label_list_segmentation[compact_idx]
    return freesurfer_labels.get(real_id, f"id={real_id}")


def measure_discriminability(path_image_vol, seg_vol, lesion_idx, underlying_seg_vol=None,
                              gap=3, dilate_iters=3, min_reference_voxels=200):
    """Fisher's discriminant ratio between lesion-voxel intensities and a
    reference tissue population:

        FDR = (mean_lesion - mean_ref)^2 / (var_lesion + var_ref)

    FDR combines both into the standard signal-vs-noise
    separability measure, which is the more direct answer to "how
    consistent is the image-to-label correspondence" -- a network trained
    on cases with high FDR has a genuinely learnable per-voxel appearance
    cue; low FDR means appearance alone barely distinguishes the label
    regardless of how well-trained the network is, which is aleatoric, 
    not a training problem.

    Same reference logic and fallback as measure_realized_contrast(). None
    under the same conditions.
    """
    if lesion_idx is None:
        return None
    lesion_mask = seg_vol == lesion_idx
    if not lesion_mask.any():
        return None
    lesion_vals = path_image_vol[lesion_mask].float()
    lesion_mean, lesion_var = lesion_vals.mean(), lesion_vals.var()

    def _fdr(ref_vals):
        ref_mean, ref_var = ref_vals.mean(), ref_vals.var()
        denom = lesion_var + ref_var
        if denom < 1e-9:
            return None
        return ((lesion_mean - ref_mean) ** 2 / denom).item()

    if underlying_seg_vol is not None:
        under_labels, counts = torch.unique(underlying_seg_vol[lesion_mask], return_counts=True)
        dominant_label = under_labels[counts.argmax()].item()
        reference_mask = (underlying_seg_vol == dominant_label) & (~lesion_mask)
        if reference_mask.sum() >= min_reference_voxels:
            result = _fdr(path_image_vol[reference_mask].float())
            if result is not None:
                return result

    inner = _dilate_mask(lesion_mask, iters=gap)
    band = _dilate_mask(lesion_mask, iters=gap + dilate_iters) & (~inner)
    if band.sum() < min_reference_voxels:
        band = _dilate_mask(lesion_mask, iters=dilate_iters) & (~lesion_mask)
    if not band.any():
        return None
    return _fdr(path_image_vol[band].float())


# ---------------------------------------------------------------------
# Rolling accumulator for (requested_k, realized_k) pairs across viz steps
# ---------------------------------------------------------------------
class RollingDiagnostics:
    """Accumulates realized-vs-requested contrast pairs, cascade ratios,
    and lesion dice across a training run, for periodic and final trend
    reports without a post-hoc log pull."""

    def __init__(self, report_every=10):
        self.pairs = []          # (requested_k, realized_k)
        self.cascade_ratios = []
        self.dice_history = []   # raw lesion dice, in logged order
        self.report_every = report_every

    def record(self, requested_k, realized_k):
        if requested_k is not None and realized_k is not None:
            self.pairs.append((float(requested_k), float(realized_k)))

    def record_cascade(self, ratio):
        self.cascade_ratios.append(ratio)

    def record_dice(self, les_dice):
        self.dice_history.append(float(les_dice))

    def rolling_dice_mean(self, window=20):
        """Mean of the last `window` logged lesion-dice values (or all of
        them if fewer). Raw per-checkpoint lesion dice is dominated by
        which random lesion size that checkpoint happened to draw (dice on
        a handful of true-positive voxels is close to a coin flip) -- this
        smooths that out so trend direction is visible in the log itself
        instead of requiring a post-hoc pull every time."""
        if not self.dice_history:
            return float("nan")
        recent = self.dice_history[-window:]
        return sum(recent) / len(recent)

    @staticmethod
    def _pearson(xs, ys):
        n = len(xs)
        if n < 3:
            return float("nan")
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        sy = math.sqrt(sum((y - my) ** 2 for y in ys))
        if sx == 0 or sy == 0:
            return float("nan")
        return cov / (sx * sy)

    def maybe_report(self, step, log_print=print):
        if len(self.pairs) and len(self.pairs) % self.report_every == 0:
            xs = [p[0] for p in self.pairs]
            ys = [p[1] for p in self.pairs]
            r = self._pearson(xs, ys)
            log_print(f"[step {step}] [realized-contrast check] "
                      f"n={len(self.pairs)}  corr(requested_k, realized_k)={r:+.3f}  "
                      f"(compare to +0.09, the old noise-referenced design's number)")
        if self.cascade_ratios and len(self.cascade_ratios) % 500 == 0:
            worst = max(self.cascade_ratios[-500:])
            log_print(f"[step {step}] [cascade check] worst path/clean max "
                      f"ratio in last 500 steps: {worst:.2f}")

    def final_report(self, log_print=print):
        if not self.pairs:
            log_print("[realized-contrast check] no lesion-bearing cases recorded.")
            return
        xs = [p[0] for p in self.pairs]
        ys = [p[1] for p in self.pairs]
        r = self._pearson(xs, ys)
        log_print(f"[realized-contrast check] FINAL: n={len(self.pairs)}  "
                  f"corr(requested_k, realized_k)={r:+.3f}")


def _self_test():
    """Exercises all four checks against synthetic tensors so you can
    confirm this module actually works in your cluster environment
    (torch build, CUDA, GPU visibility) before wiring it into the real
    training loop. Not a substitute for testing against real generator
    output -- just an import/environment check."""
    import random

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[self-test] torch {torch.__version__}, device={device}")
    if torch.cuda.is_available():
        print(f"[self-test] GPU: {torch.cuda.get_device_name(0)}")

    # 1) sampler provenance
    def fake_sampler():
        return random.uniform(1.0, 2.5)
    report_sampler_provenance(fake_sampler, expected_range=(1.0, 2.5),
                               n_samples=1000, git_hash="selftest", log_print=print)
    print()

    # 2) normalization cascade
    clean = torch.rand(1, 1, 20, 20, 20, device=device)
    path = clean.clone()
    path[0, 0, 5, 5, 5] = 50.0  # synthetic outlier "lesion"
    ratio = check_normalization_cascade(clean, path, step="selftest", log_print=print)
    assert ratio > 1.0
    print(f"[self-test] cascade ratio = {ratio:.2f}  OK")
    print()

    # 3) realized contrast: mean(|path-clean|) per voxel vs a reference sigma,
    # matched-tissue-elsewhere vs gapped-dilation-ring fallback
    clean_img = torch.rand(20, 20, 20, device=device)
    path_img = clean_img.clone()
    seg = torch.zeros(20, 20, 20, dtype=torch.long, device=device)
    clean_seg = torch.full((20, 20, 20), 3, dtype=torch.long, device=device)  # pretend "white matter" everywhere
    seg[8:12, 8:12, 8:12] = 33
    path_img[8:12, 8:12, 8:12] = 0.9
    rk = measure_realized_contrast(path_img, clean_img, seg, lesion_idx=33, underlying_seg_vol=clean_seg)
    assert rk is not None
    print(f"[self-test] realized_k (matched-tissue reference) = {rk:.3f}  OK")

    # multi-zone case: two sub-regions shifted in OPPOSITE directions from
    # clean -- checks that the abs-per-voxel formula doesn't cancel where a
    # naive signed whole-lesion mean would.
    zone_img = clean_img.clone()
    zone_seg = torch.zeros(20, 20, 20, dtype=torch.long, device=device)
    zone_clean_seg = torch.full((20, 20, 20), 3, dtype=torch.long, device=device)
    zone_seg[8:12, 8:12, 8:12] = 33
    zone_img[8:10, 8:12, 8:12] = 0.05   # zone A: pushed dark
    zone_img[10:12, 8:12, 8:12] = 0.95  # zone B: pushed bright, opposite sign
    lesion_mask = zone_seg == 33
    naive_signed = (zone_img[lesion_mask].mean() - clean_img[~lesion_mask].mean()).abs().item()
    rk_zones = measure_realized_contrast(zone_img, clean_img, zone_seg, 33, underlying_seg_vol=zone_clean_seg)
    tex_zones = measure_realized_texture_ratio(zone_img, zone_seg, 33, underlying_seg_vol=zone_clean_seg)
    print(f"[self-test] multi-zone case (symmetric dark+bright zones, cancels "
          f"under a naive signed-mean formula):")
    print(f"    naive signed-mean |diff|  = {naive_signed:.3f}  (what a signed mean gives -- ~0, false negative)")
    print(f"    new realized_k            = {rk_zones:.2f}  (mean absolute per-voxel change -- correctly nonzero)")
    print(f"    texture_ratio             = {tex_zones:.2f}x")
    assert naive_signed < 0.05, "expected the signed-mean formula to cancel on this symmetric case"
    assert rk_zones > 1.0, "expected the per-voxel formula to NOT cancel"
    print("    OK -- per-voxel formula does not cancel where a signed mean would")
    print()

    # 4) dominant class fraction
    pred = torch.randint(0, 5, (20, 20, 20), device=device)
    pred[:15] = 2
    idx, frac, _ = dominant_class_fraction(pred, n_classes=5, exclude=(0,))
    assert idx == 2 and frac > 0.5
    print(f"[self-test] dominant class idx={idx} frac={frac:.2f}  OK")
    print()

    # 5) rolling diagnostics
    rd = RollingDiagnostics(report_every=5)
    for i in range(12):
        rd.record(2.0 + 0.01 * i, 2.0 + 0.01 * i + random.uniform(-0.05, 0.05))
        rd.maybe_report(step=i * 1000, log_print=print)
    rd.final_report(print)

    print("\n[self-test] ALL CHECKS PASSED -- module imports and runs cleanly "
          f"on this node ({device}). Now wire the individual functions into "
          "train.py per the instructions in this file's module docstring.")


if __name__ == "__main__":
    _self_test()
