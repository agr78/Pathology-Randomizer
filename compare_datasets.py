import os, glob, sys, torch, argparse, csv, re
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np

# Makes the networks.py import below resolve regardless of whether this
# file lives in the repo root or a subdirectory such as utils/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks import (SynthSegUNet, detect_norm, load_checkpoint,
                      detect_regression_channels, detect_probability_channels,
                      detect_attention, detect_attention_dense, detect_seg_prior, detect_reg_head, detect_tissue_rel, detect_tissue_refs, detect_tissue_vec, detect_tissue_pool, detect_tissue_unroll, detect_reg_arch, detect_tissue_soft)

# Compact index of 'Lesion' in this list, not the FreeSurfer id 33: the
# network's output channels are compact indices 0-55, and compact index 33
# in this list is FreeSurfer id 47, a different structure entirely.
LABEL_LIST_SEGMENTATION = [0, 11, 12, 13, 16, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                           43, 44, 46, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 17, 47, 49, 51,
                           53, 55, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 48, 50,
                           52, 54, 56]
LESION_COMPACT_IDX = LABEL_LIST_SEGMENTATION.index(33)

# Compact indices for the extracerebral block (raw FreeSurfer ids 34-46,
# skipping 45 -- always killed by generator.py's labels_to_kill=[45] and
# so never assigned a compact index at all). Computed from
# LABEL_LIST_SEGMENTATION directly rather than hand-counting positions, to
# stay correct if that list ever changes.
EXTRACEREBRAL_COMPACT_IDX = [LABEL_LIST_SEGMENTATION.index(raw_id)
                            for raw_id in range(34, 47)
                            if raw_id in LABEL_LIST_SEGMENTATION]

TARGET_LABELS = {"whole": (1, 2, 3), "core": (1, 3), "enhancing": (3,)}

# Module level so run_inference, the Dice printout and eval/ share one value.
REGRESSION_LESION_THRESHOLD = 0.05

# _reg.nii.gz stores (reg + 1) / 2, so reg = 0 is 0.5 and an array of zeros
# decodes to reg = -1 -- a maximal DARK response. Anything filling voxels the
# network did not produce (outside the head bbox; outside a storage mask) must
# use this, not np.zeros. One constant so the writer and any masker agree.
REG_STORED_ZERO = 0.5
REG_INT16_SLOPE = 1.0 / 65534.0      # stored value = int16 * slope + inter
REG_INT16_INTER = 32767.0 / 65534.0


def cerebral_mask(seg):
    """Voxels the seg head calls brain -- not background, not extracerebral.

    Lookup table rather than np.isin: isin runs one equality pass per label,
    so 13 labels over 8.7M voxels cost 0.47 s/case against 0.06 s here, for a
    bit-identical result. Measured -- it was 45% of a full-derivative pass.
    """
    # Sized by the label list, not seg.max(): a segmentation with no label
    # above the extracerebral indices made the table too short (IndexError,
    # BraTS FLAIR sub-01157, 2026-09-17).
    lut = np.zeros(max(len(LABEL_LIST_SEGMENTATION), int(seg.max()) + 1), dtype=bool)
    lut[[0] + EXTRACEREBRAL_COMPACT_IDX] = True
    return ~lut[seg.astype(np.int32)]


def lesion_mask_from(reg_path, seg_path):
    """The |reg| overwrite mask, rebuilt from _reg + _seg.

    _lesion.nii.gz is no longer written -- it held exactly this. _reg stores
    (reg+1)/2, so the round-trip loses the last bit and a voxel sitting on the
    threshold can flip; measured at 0-23 voxels per case.
    """
    reg = 2.0 * np.asarray(nib.load(reg_path).dataobj, dtype=np.float32) - 1.0
    seg = np.asarray(nib.load(seg_path).dataobj)
    return (np.abs(reg) > REGRESSION_LESION_THRESHOLD) & cerebral_mask(seg)

# Per-dataset ground-truth layout, target labels, and which T1 to use.
# eval_lesion_dice.py imports these from here rather than defining its own.
#
# WMH is the awkward one: its masks live on the FLAIR grid (e.g. 132x256x83
# at 1.2 x 0.98 x 3.0 mm), which is T1.nii.gz -- the copy resampled into
# FLAIR space -- NOT 3DT1.nii.gz, the native isotropic acquisition
# find_t1_files() returns by default. Using 3DT1 makes every case fail the
# grid-match check.
DATASETS = {
    "BraTS": dict(t1_name=None, labels=None,   # labels from --target
                  gt=lambda t1, root, sub: os.path.join(
                      root, "derivatives", "manual_segmentation", sub, f"{sub}_manual_dseg.nii.gz")),
    "ATLAS": dict(t1_name=None, labels=(1,),
                  gt=lambda t1, root, sub: os.path.join(
                      root, "derivatives", "manual_segmentation", sub, "anat",
                      f"{sub}_manual_stroke_dseg.nii.gz")),
    # label 2 is 'other pathology' and is excluded from evaluation by the
    # challenge's own protocol, so only label 1 counts.
    "WMH": dict(t1_name="T1.nii.gz", labels=(1,),
                gt=lambda t1, root, sub: os.path.join(
                    os.path.dirname(os.path.dirname(t1)), "derivatives",
                    "manual_segmentations", sub, "wmh.nii.gz")),
}


def dataset_key(name):
    for k in DATASETS:
        if name.upper().startswith(k.upper()):
            return k
    return "BraTS"


def dice(a, b):
    """Dice between two boolean masks. Returns None if BOTH are empty --
    scoring that as 1.0 would reward a model for finding nothing in a case
    with no tumour, and averaging those in would inflate the result."""
    denom = a.sum() + b.sum()
    if denom == 0:
        return None
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def normalize_like_training(img_tensor, mask=None):
    """Scale to [0, 1] via SynthSeg's own inference-time normalization:
    clip to [1st, 99th] percentile, then rescale that clipped range to
    [0, 1]. Must match train.py's normalize_per_sample EXACTLY -- the
    network never sees any other range, at train or inference time.

    This used to clip only the upper end (99.5th percentile), leaving the
    lower end as a plain min -- an approximation of SynthSeg's own
    published normalization (clip to [1st, 99th] on both ends), not an
    exact match. Now exact, and identical to train.py's own formula rather
    than an independently-maintained approximation of it.
    """
    img = img_tensor.clamp(min=0)

    # mask (optional, --norm_mask): take the percentiles inside a brain mask
    # instead of the whole >0 foreground. Diagnostic for scans whose p99 sits
    # in scalp (non-stripped FLAIR): training normalizes on the full
    # foreground, so this is deliberately NOT the training frame.
    fg = img[(img > 0) & mask] if mask is not None else img[img > 0]
    if fg.numel() > 0:
        p_lo, p_hi = torch.quantile(fg, torch.tensor([0.01, 0.99], device=fg.device, dtype=fg.dtype))
    else:
        p_lo, p_hi = torch.tensor(0.0, device=img.device), torch.tensor(1.0, device=img.device)

    img = img.clamp(min=p_lo, max=p_hi) - p_lo
    scale = (p_hi - p_lo).clamp(min=1e-7)
    return (img / scale).clamp(0, 1)


def to_ras(nii):
    """Reorient a volume to RAS, returning (data, inverse-orientation).

    The generator loads training volumes with nibabel and crops; it never
    reorients. brain_mris_QCed is RAS throughout, so that is the only axis
    order the network has ever seen, and a convolutional network is not
    invariant to axis permutation -- training's flipping=True covers
    left-right flips only.

    Axis codes: ATLAS RAS (already matches), BraTS LPS (L-R and A-P both
    flipped), WMH T1/FLAIR LPS, WMH 3DT1 PIR (axis order permuted).

    The inverse is returned so the prediction can be written back onto the
    original grid, keeping it comparable with ground truth stored that way.
    """
    from nibabel.orientations import (io_orientation, apply_orientation,
                                      ornt_transform, axcodes2ornt)
    data = np.asanyarray(nii.dataobj)
    while data.ndim > 3:
        data = data[..., 0]
    ornt = io_orientation(nii.affine)
    inv = ornt_transform(axcodes2ornt("RAS"),
                         axcodes2ornt(nib.aff2axcodes(nii.affine)))
    return apply_orientation(data, ornt).astype(np.float32), inv


def from_ras(arr, inv_ornt):
    """Undo to_ras, putting a prediction back on the original grid."""
    from nibabel.orientations import apply_orientation
    return apply_orientation(arr, inv_ornt)


def _find_t1_files_impl(data_dir, dataset, modality="3DT1"):
    """Locate the T1 volumes for a dataset, handling its directory layout.

    Layouts differ enough that one glob doesn't cover them:
      BraTS/ATLAS  {data_dir}/sub-XXXXX/anat/sub-XXXXX_T1w.nii.gz
      EPAD         {data_dir}/EPAD_bids/sub-XXXXX/ses-YY/anat/sub-XXXXX_ses-YY_T1w.nii.gz
      WMH          {data_dir}/{SITE}/sub-XX/3DT1.nii.gz     (no anat/, no BIDS suffix)

    WMH ships three volumes per subject, selected with --modality:
      3DT1   native isotropic T1, closest to what the network trained on
      T1     the same T1 resampled into FLAIR space (1.2 x 0.98 x 3.0 mm)
      FLAIR  the sequence the challenge ground truth was actually drawn on

    FLAIR is worth running even though the network never saw FLAIR contrast.
    WMH lesions are hyperintense on FLAIR and hypo- or isointense on T1, which
    is why this dataset is the hardest of the three: on T1 the thing being
    segmented is barely visible. Comparing the two says whether the failure is
    contrast (FLAIR much better) or something else (both equally poor).

    Note FLAIR and the FLAIR-space T1 have 3mm slices, so resampling to 1mm
    interpolates through-plane detail that was never acquired.
    """
    if dataset.startswith("WMH"):
        if modality not in ("3DT1", "T1", "FLAIR"):
            raise SystemExit(f"--modality for WMH must be 3DT1, T1 or FLAIR (got {modality})")
        files = glob.glob(os.path.join(data_dir, "**", f"{modality}.nii.gz"), recursive=True)
    else:
        # Restricted to /anat/ so derivative or processed copies elsewhere in
        # the tree aren't picked up. The *_T1w suffix already excludes FLAIR,
        # T2w and T2star, so multi-contrast sessions only contribute their T1.
        files = [
            p for p in glob.glob(os.path.join(data_dir, "**", "*_T1w.nii.gz"), recursive=True)
            if os.sep + "anat" + os.sep in p
        ]
    return sorted(files)


def list_cases(data_dir, dataset, modality=None):
    """(subject, t1_path, gt_path) for every case with a ground-truth mask.

    Shared by eval_lesion_dice.py and this file's own --single_model
    ground-truth scoring, so both use identical subject-name and
    ground-truth-path logic.
    """
    key = dataset_key(dataset)
    cfg = DATASETS[key]

    # modality only means anything for WMH (3DT1 / T1 / FLAIR) -- gating it
    # this way means a caller's own non-WMH default for --modality (e.g.
    # compare_datasets.py's "3DT1", used elsewhere for run_inference) can't
    # silently override BraTS/ATLAS's find_t1_files()-based lookup here.
    name = modality if (key == "WMH" and modality) else cfg["t1_name"]
    # --modality carries a label ("FLAIR", "3DT1"); the files on disk are
    # <label>.nii.gz. Without this the glob matches nothing, every case drops
    # out silently, and scoring reports "0 cases with ground truth".
    if name and key == "WMH" and not name.endswith(".nii.gz"):
        name = f"{name}.nii.gz"
    if name:
        t1s = sorted(glob.glob(os.path.join(data_dir, "**", name), recursive=True))
    else:
        t1s = find_t1_files(data_dir, dataset)
    t1s = _apply_subject_filter(t1s, dataset)   # --subjects (WMH globs here without find_t1_files)

    out = []
    for t1 in t1s:
        if key == "WMH":
            sub = os.path.basename(os.path.dirname(t1))
        else:
            sub = os.path.basename(t1).split("_")[0]
        gt = cfg["gt"](t1, data_dir, sub)
        if os.path.isfile(gt):
            label = sub if key != "WMH" else f"{os.path.basename(os.path.dirname(os.path.dirname(t1)))}_{sub}"
            out.append((label, t1, gt))
    return out


def find_t1_files(data_dir, dataset, modality="3DT1"):
    """BIDS datasets (BraTS, ATLAS) ship other contrasts next to the T1w with
    the same naming; --modality FLAIR / T2w / T1ce swaps the suffix. The
    network is contrast-agnostic, so BraTS on FLAIR (tumour bright) is the
    direct test of the T1-dark hypothesis."""
    files = _find_t1_files_impl(data_dir, dataset, modality)
    if not dataset.startswith("WMH") and modality not in ("3DT1", "T1", "T1w"):
        swapped = [f.replace("_T1w.nii.gz", f"_{modality}.nii.gz") for f in files]
        files = [f for f in swapped if os.path.isfile(f)]
    return _apply_subject_filter(files, dataset)


SUBJECT_FILTER = None   # --subjects: only these cases are inferred and scored (None = every case)


def _subject_ids(path, dataset):
    """The names a --subjects token may use for this file: 'sub-536' for BIDS datasets;
    'sub-15' (any site) or 'Utrecht_sub-15' for WMH, whose sites reuse subject numbers."""
    if dataset.startswith("WMH"):
        sub = os.path.basename(os.path.dirname(path))
        site = os.path.basename(os.path.dirname(os.path.dirname(path)))
        return {sub, f"{site}_{sub}"}
    return {os.path.basename(path).split("_")[0]}


def _apply_subject_filter(files, dataset):
    if not SUBJECT_FILTER:
        return files
    return [f for f in files if _subject_ids(f, dataset) & SUBJECT_FILTER]


def output_name_for(t1_path, data_dir, dataset, modality="3DT1"):
    """Build a unique output filename for a T1 path.

    BIDS datasets carry subject (and session) in the filename, so the
    basename is already unique. WMH names every file 3DT1.nii.gz, so the
    identity lives entirely in the directory path ({SITE}/sub-XX) and has to
    be folded into the name or every subject overwrites the last.
    """
    if dataset.startswith("WMH"):
        # Modality in the name: without it, segmenting 3DT1 and then FLAIR
        # would write to the same file and the second would silently replace
        # the first.
        rel = os.path.relpath(os.path.dirname(t1_path), data_dir)
        return rel.replace(os.sep, "_") + f"_{modality}_seg.nii.gz"
    base = os.path.basename(t1_path)
    m = re.search(r"_(T1w|FLAIR|T2w|T1ce)\.nii\.gz$", base)
    if m and m.group(1) != "T1w":
        return base[:m.start()] + f"_{m.group(1)}_seg.nii.gz"
    return base.replace("_T1w.nii.gz", "_seg.nii.gz")


def head_bbox(img_data, thresh_frac=0.02, margin=8):
    """Bounding box of the head, from a low intensity threshold.

    Threshold is a fraction of the robust max rather than >0, since real
    scanner data has nonzero noise in air -- >0 would return the whole
    volume and do nothing.
    """
    hi = np.percentile(img_data[img_data > 0], 99) if (img_data > 0).any() else 0.0
    fg = img_data > (thresh_frac * hi)
    if not fg.any():
        return tuple(slice(0, s) for s in img_data.shape)
    nz = np.argwhere(fg)
    lo = np.maximum(nz.min(0) - margin, 0)
    up = np.minimum(nz.max(0) + 1 + margin, np.array(img_data.shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, up))


def gradient_magnitude_3d_np(x):
    """Same finite-difference gradient magnitude as
    diagnose_edge_shortcut.py's gradient_magnitude_3d -- reimplemented
    for a plain numpy (D,H,W) array since this file works in numpy after
    inference rather than staying in torch. Kept as a literal
    reimplementation rather than importing across files, since
    diagnose_edge_shortcut.py isn't otherwise a dependency of this one and
    the two together being unnecessarily coupled isn't worth avoiding one
    ~10-line duplicate.
    """
    xt = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    gx = xt[:, :, 2:, :, :] - xt[:, :, :-2, :, :]
    gy = xt[:, :, :, 2:, :] - xt[:, :, :, :-2, :]
    gz = xt[:, :, :, :, 2:] - xt[:, :, :, :, :-2]
    gx = gx[:, :, :, 1:-1, 1:-1]
    gy = gy[:, :, 1:-1, :, 1:-1]
    gz = gz[:, :, 1:-1, 1:-1, :]
    grad = torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-12)
    return grad.squeeze(0).squeeze(0).numpy()


def pearson_corr_np(a, b):
    """Pearson correlation between two flattened numpy arrays. NaN if
    either has ~zero variance (e.g. a mask too small/uniform to be
    meaningful), rather than dividing by ~0 and returning a meaningless
    huge or undefined value silently."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum()) * np.sqrt((b**2).sum())
    if denom < 1e-8:
        return float("nan")
    return float((a * b).sum() / denom)


def partial_corr_np(r_xy, r_xz, r_yz):
    """Partial correlation of X and Y controlling for Z, from the three
    pairwise correlations. Works identically for a continuous/continuous
    pair or a continuous/binary pair (point-biserial correlation is just
    Pearson correlation applied to a 0/1 variable, so the same formula
    holds) -- used below with a binary extracerebral-membership variable.
    """
    denom = np.sqrt(max((1 - r_xz**2) * (1 - r_yz**2), 1e-12))
    return (r_xy - r_xz * r_yz) / denom


def analyze_intensity_vs_extracerebral(reg_crop, pred_crop, work, min_voxels=100):
    """Disentangles whether |reg_out| is driven by raw intensity (a "fires
    on very bright/dark voxels" bias) versus genuine extracerebral-tissue
    membership -- the two are confounded on real data, since skull/marrow
    fat is often bright on T1, exactly coinciding with the extracerebral
    region analyze_extracerebral_edge_correlation measures. A high
    extracerebral correlation could mean the tissue TYPE is the real
    driver, or it could mean intensity is the real driver and
    extracerebral tissue just happens to usually be at the bright end --
    the same confound the edge-vs-genuine-pathology-signal question had,
    solved the same way: partial correlation.

    Verified directly on synthetic data with a known answer: a purely
    intensity-driven signal (membership has no independent effect) shows
    raw corr(reg_out, membership)=0.591 -- meaningful-looking on its own --
    collapsing to a partial correlation of 0.002 once intensity is
    controlled for. A purely membership-driven signal (intensity has no
    independent effect) shows the exact reverse pattern.

    Pools ALL voxels together rather than splitting by predicted tissue
    type, since disentangling two variables that are correlated WITH EACH
    OTHER across the volume is the whole point -- a within-group split
    would remove exactly the covariation being tested.
    """
    normed = normalize_like_training(torch.from_numpy(work).float()).numpy()

    reg_flat = np.abs(reg_crop).flatten()
    intensity_flat = normed.flatten()
    membership_flat = np.isin(pred_crop, EXTRACEREBRAL_COMPACT_IDX).astype(np.float64).flatten()

    n_total = len(reg_flat)
    n_membership = int(membership_flat.sum())
    # Both groups need real representation, or membership has ~no variance
    # to correlate with anything at all.
    if n_total < min_voxels or n_membership < min_voxels or n_membership > n_total - min_voxels:
        return dict(r_reg_membership=float("nan"), r_reg_intensity=float("nan"),
                   r_intensity_membership=float("nan"),
                   partial_membership_given_intensity=float("nan"),
                   partial_intensity_given_membership=float("nan"), n=n_total)

    r_reg_membership = pearson_corr_np(reg_flat, membership_flat)
    r_reg_intensity = pearson_corr_np(reg_flat, intensity_flat)
    r_intensity_membership = pearson_corr_np(intensity_flat, membership_flat)

    return dict(
        r_reg_membership=r_reg_membership,
        r_reg_intensity=r_reg_intensity,
        r_intensity_membership=r_intensity_membership,
        partial_membership_given_intensity=partial_corr_np(
            r_reg_membership, r_reg_intensity, r_intensity_membership),
        partial_intensity_given_membership=partial_corr_np(
            r_reg_intensity, r_reg_membership, r_intensity_membership),
        n=n_total,
    )


def analyze_extracerebral_edge_correlation(reg_crop, pred_crop, work, min_voxels=100):
    """Correlates |reg_crop| with local image gradient magnitude, computed
    SEPARATELY within voxels the segmentation head itself predicts as
    extracerebral (EXTRACEREBRAL_COMPACT_IDX) versus predicted brain
    tissue (foreground, excluding extracerebral).

    Directly comparable to diagnose_edge_shortcut.py's Test 1, which runs
    this same correlation on synthetic validation data and found r=0.220
    in-distribution -- that number is a baseline, not evidence either way
    about real-scan behavior, since synthetic validation cases share the
    training distribution by construction. This runs the identical logic
    on REAL data instead, split by predicted tissue type specifically to
    test whether extracerebral tissue -- killed 50% of the time in
    training and given a random, uncorrelated intensity from the same GMM
    table as every other label when present (see generator.py's
    random_kill_labels) -- shows a materially worse correlation than the
    brain-tissue region or the in-distribution baseline, which would
    point at the sim-to-real gap on that specific tissue rather than a
    general flaw in the regression head itself.

    work is the pre-normalization, head-cropped image at the SAME
    resolution reg_crop/pred_crop occupy (before any resample-back is
    needed) -- computing gradient from the raw image rather than the
    normalized tensor, since gradient magnitude's own scale is irrelevant
    to a correlation coefficient, only its spatial pattern is.
    """
    grad = gradient_magnitude_3d_np(work)
    # gradient_magnitude_3d_np crops 1 voxel off each edge (see its own
    # finite-difference construction) -- crop reg_crop/pred_crop to match.
    reg_cropped = reg_crop[1:-1, 1:-1, 1:-1]
    pred_cropped = pred_crop[1:-1, 1:-1, 1:-1]

    extracerebral_mask = np.isin(pred_cropped, EXTRACEREBRAL_COMPACT_IDX)
    brain_mask = (pred_cropped != 0) & ~extracerebral_mask

    result = {}
    for name, mask in [("extracerebral", extracerebral_mask), ("brain", brain_mask)]:
        n = int(mask.sum())
        if n < min_voxels:
            result[name] = dict(r=float("nan"), n=n)
            continue
        result[name] = dict(r=pearson_corr_np(np.abs(reg_cropped[mask]), grad[mask]), n=n)
    return result


def bn_batch_mode(model):
    """--bn_mode batch: BatchNorm normalizes with the statistics of the image
    being inferred; everything else stays in eval. Training forwarded each
    rendering's (clean, path) pair as one batch, so the weights only ever saw
    batch statistics. momentum 0 leaves the running buffers untouched."""
    n = 0
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train()
            m.momentum = 0.0
            n += 1
    return n


def load_mask_derivative(output_dir, deriv, filename, bbox):
    """Brain mask from a sibling derivative (e.g. synthstrip_b0): same name as
    the output (<name>_seg.nii.gz), same RAS + bbox as the image.
    Returns (bool array | None, "direct" | "fallback" | "missing")."""
    mp = os.path.join(os.path.dirname(output_dir), deriv, filename)
    status = "direct"
    if not os.path.isfile(mp):
        # BraTS: one T1w-named synthstrip mask serves every co-registered
        # modality (as dice_by_mask.py and mask_reg_maps.py do); WMH has a
        # mask per modality and never takes this branch.
        alt = os.path.join(os.path.dirname(output_dir), deriv,
                           re.sub(r"_(FLAIR|T1w?)_seg\.nii\.gz$", "_seg.nii.gz", filename))
        if os.path.isfile(alt):
            mp, status = alt, "fallback"
    if not os.path.isfile(mp):
        return None, "missing"
    m_data, _ = to_ras(nib.load(mp))
    return m_data[bbox] > 0, status


def run_inference(ckpt_path, data_dir, output_dir, dataset, overwrite=False, mask=False,
                  crop=True, target_mm=1.0, modality="3DT1", diagnose_edges=False, norm_mask=None,
                  bn_mode="running", strip_mask=None, tissue_pool=None, zero_refs=False):
    """Segment each T1, following SynthSeg's own inference recipe.

    SynthSeg resamples the input to 1mm isotropic, normalizes intensities to
    [0,1], and runs the WHOLE volume through in a single forward pass -- it
    does not tile. Its --crop flag exists only to fit large volumes in GPU
    memory, and by default the whole image is processed.

    What has to match between training and inference is the VOXEL SPACING,
    not the array shape: the net is fully convolutional, so it accepts any
    size divisible by 32, but its features are tuned to structures at the
    spacing it trained on (1mm). A 0.8mm clinical scan run without
    resampling presents every structure ~25% larger than the net expects.

    diagnose_edges: also runs analyze_extracerebral_edge_correlation on
    every case with a regression head, printing a per-case line and a
    final summary across the whole dataset -- see that function's own
    docstring for what this tests and why.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint["model_state_dict"] if (isinstance(checkpoint, dict)
                                               and "model_state_dict" in checkpoint) else checkpoint
    norm = detect_norm(state)
    reg_ch = detect_regression_channels(state)
    # Detected, not assumed: a checkpoint with a probability head carries
    # probability_conv.* keys that a model built without one rejects outright.
    prob_ch = detect_probability_channels(state)
    # Architecture variants that change the parameter set are detected the
    # same way, from the weights: a checkpoint built without them rejects the
    # state_dict outright (size mismatch on regression_conv / missing att4.*).
    attention = detect_attention(state)
    att_dense = detect_attention_dense(state)
    seg_prior = detect_seg_prior(state)
    print(f"[Inference] checkpoint uses {norm.__name__} with {reg_ch} regression channels"
          + (f" and {prob_ch} probability channels" if prob_ch else "")
          + (", skip attention" if attention else "") + (" (dense gates)" if att_dense else "")
          + (", seg prior into regression" if seg_prior else ""))
    n_out = state["final_conv.weight"].shape[0]                     # 56, or 58 with --choroid_target (compact 56/57 = L/R choroid plexus)
    if n_out == 58 and 57 not in LABEL_LIST_SEGMENTATION:
        LABEL_LIST_SEGMENTATION.extend([57, 58]); print("[Inference] 58-class checkpoint: choroid plexus classes appended")
    model = SynthSegUNet(out_channels=n_out, norm=norm, regression_channels=reg_ch,
                         probability_channels=prob_ch, attention=attention, attention_dense=att_dense,
                         seg_prior=seg_prior, reg_head=detect_reg_head(state),   # head variants (2026-09-23); old checkpoints: linear, no rel channel
                         tissue_rel=detect_tissue_rel(state), tissue_refs=detect_tissue_refs(state), tissue_vec=detect_tissue_vec(state), tissue_pool=tissue_pool or detect_tissue_pool(state), tissue_unroll=detect_tissue_unroll(state), tissue_soft=detect_tissue_soft(state),
                         **dict(zip(("reg_hidden", "reg_depth", "reg_kernel"), detect_reg_arch(state))) if detect_reg_head(state) == "mlp" else {}).to(device)
    if tissue_pool and tissue_pool != detect_tissue_pool(state):
        # re-inference with other pools than the head was trained on: same channel count, only the marker buffer differs
        missing = model.load_state_dict(state, strict=False).missing_keys
        assert set(missing) <= {"tissue_pool_flag"}, f"unexpected missing keys {missing}"
        print(f"[tissue_pool] override: {detect_tissue_pool(state)} -> {tissue_pool} (head trained on the former; a diagnostic, not the trained configuration)")
    else:
        model.load_state_dict(state)
    if zero_refs:
        # dependency diagnostic (2026-10-05): the head runs with its WM/cortex/CSF reference channels zeroed, own-class channel kept.
        # How much the maps change says how much the trained head leans on the references; not the trained configuration.
        assert getattr(model, "tissue_rel", False) and getattr(model, "tissue_refs", 0), "--zero_refs needs a checkpoint with reference channels"
        _tr_forward, _n_refs = model.tissue_relative.forward, int(model.tissue_refs)
        def _forward_zero_refs(x, logits, exclude=None):
            rel = _tr_forward(x, logits, exclude); rel[:, 1:1 + _n_refs] = 0; return rel
        model.tissue_relative.forward = _forward_zero_refs
        print(f"[zero_refs] {_n_refs} reference channels zeroed at inference (own-class channel kept); use --out_suffix so the maps do not overwrite the trained configuration's")
    model.eval()
    if bn_mode == "batch":
        print(f"[bn_mode] batch statistics in {bn_batch_mode(model)} BatchNorm layers")

    t1_files = find_t1_files(data_dir, dataset, modality)
    if not t1_files:
        print(f"[Inference] WARNING: no T1 files found under {data_dir}")
    print(f"\n[Inference] Running {len(t1_files)} files for checkpoint: {os.path.basename(ckpt_path)}")

    edge_diag_results = []
    intensity_diag_results = []
    mask_counts = {"strip_mask": {}, "norm_mask": {}}   # status -> n, printed at the end

    for t1_path in t1_files:
        filename = output_name_for(t1_path, data_dir, dataset, modality)
        out_pred_path = os.path.join(output_dir, filename)

        # diagnose_edges needs a fresh inference pass regardless of
        # whether the output file gets re-saved -- without this, a case
        # already processed by an earlier run would be silently skipped
        # entirely, and diagnose_edges could not be run on it without
        # --overwrite forcing a full re-save too.
        if os.path.exists(out_pred_path) and not overwrite and not diagnose_edges:
            continue

        nii = nib.load(t1_path)
        # RAS before anything else -- see to_ras. BraTS is LPS, WMH 3DT1 is PIR,
        # and the network has only ever seen RAS.
        img_data, inv_ornt = to_ras(nii)
        zooms = np.array(nii.header.get_zooms()[:3], dtype=np.float32)

        # Crop to the head. Not required by SynthSeg, but InstanceNorm3d
        # derives its statistics from the input at inference time (no stored
        # running stats), so a volume that is mostly air normalizes very
        # differently from one that is mostly tissue.
        if crop:
            bbox = head_bbox(img_data)
        else:
            bbox = tuple(slice(0, s) for s in img_data.shape)
        work = img_data[bbox]

        if strip_mask:
            # zero outside the brain BEFORE the unchanged training rule (percentiles
            # over img > 0) runs; a copy, so --mask below still sees the raw image
            m, status = load_mask_derivative(output_dir, strip_mask, filename, bbox)
            mask_counts["strip_mask"][status] = mask_counts["strip_mask"].get(status, 0) + 1
            if m is not None:
                work = np.where(m, work, np.float32(0))
            else:
                print(f"  [strip_mask] no mask for {filename}; image left unstripped")

        img_tensor = torch.from_numpy(work).to(device).unsqueeze(0).unsqueeze(0)
        norm_m = None
        if norm_mask:
            m, status = load_mask_derivative(output_dir, norm_mask, filename, bbox)
            mask_counts["norm_mask"][status] = mask_counts["norm_mask"].get(status, 0) + 1
            if m is not None:
                norm_m = torch.from_numpy(m).to(device).unsqueeze(0).unsqueeze(0)
            else:
                print(f"  [norm_mask] no mask for {filename}; foreground percentiles used")
        img_tensor = normalize_like_training(img_tensor, mask=norm_m)

        # Resample to the training spacing. SynthSeg skips this when the scan
        # is already within [0.95, 1.05] mm, which avoids a pointless
        # interpolation on data that is effectively already isotropic 1mm.
        cropped_shape = tuple(img_tensor.shape[2:])
        needs_resample = np.any(np.abs(zooms - target_mm) > 0.05 * target_mm)
        if needs_resample:
            scaled = np.maximum(np.round(np.array(cropped_shape) * zooms / target_mm), 1).astype(int)
            img_tensor = F.interpolate(img_tensor, size=tuple(scaled),
                                       mode='trilinear', align_corners=False)

        # Pad to a multiple of 32 (4 pooling levels, plus headroom)
        in_shape = img_tensor.shape[2:]
        pads = [(32 - s % 32) % 32 for s in in_shape]
        img_padded = F.pad(img_tensor, (0, pads[2], 0, pads[1], 0, pads[0]))

        with torch.no_grad(), torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
            outputs = model(img_padded)
            
            # Handle the tuple return if a regression head exists; a
            # probability head adds a third element after it.
            if reg_ch > 0:
                preds, reg = outputs[0], outputs[1]
            else:
                preds, reg = outputs, None

            preds = preds[:, :, :in_shape[0], :in_shape[1], :in_shape[2]]
            if reg is not None:
                reg = reg[:, :, :in_shape[0], :in_shape[1], :in_shape[2]]

            if needs_resample:
                preds = F.interpolate(preds.float(), size=cropped_shape,
                                      mode='trilinear', align_corners=False)
                if reg is not None:
                    reg = F.interpolate(reg.float(), size=cropped_shape,
                                        mode='trilinear', align_corners=False)
                    
            pred_crop = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy().astype(np.int32)

            if reg is not None:
                # .float(): reg comes out of autocast as float16 (CUDA) or
                # bfloat16 (CPU) -- .numpy() has no bfloat16 support at
                # all, so this was a latent gap that only ever surfaced
                # when testing without a GPU. Harmless on CUDA, since
                # float16 already converts fine, but explicit now instead
                # of relying on which device happened to be running.
                reg_crop = reg.squeeze(0).squeeze(0).float().cpu().numpy()

                if diagnose_edges:
                    edge_result = analyze_extracerebral_edge_correlation(reg_crop, pred_crop, work)
                    edge_result["case"] = os.path.basename(t1_path)
                    edge_diag_results.append(edge_result)
                    ex, br = edge_result["extracerebral"], edge_result["brain"]
                    print(f"  [edge-diag] {os.path.basename(t1_path)}: "
                          f"extracerebral r={ex['r']:+.3f} (n={ex['n']})   "
                          f"brain r={br['r']:+.3f} (n={br['n']})")

                    iv_result = analyze_intensity_vs_extracerebral(reg_crop, pred_crop, work)
                    iv_result["case"] = os.path.basename(t1_path)
                    intensity_diag_results.append(iv_result)
                    print(f"  [intensity-diag] {os.path.basename(t1_path)}: "
                          f"raw corr(reg,membership)={iv_result['r_reg_membership']:+.3f}  "
                          f"raw corr(reg,intensity)={iv_result['r_reg_intensity']:+.3f}  "
                          f"partial(membership|intensity)={iv_result['partial_membership_given_intensity']:+.3f}  "
                          f"partial(intensity|membership)={iv_result['partial_intensity_given_membership']:+.3f}")

                # Fixed rescale of the KNOWN (-1, 1) tanh range to [0, 1],
                # not a per-volume min/max. Per-volume min-max would hide
                # absolute severity: a patient with subtle pathology (true
                # peak ~0.1) and one with severe pathology (true peak ~0.8)
                # would both get rescaled to show 1.0 at their own peak,
                # making the saved maps look equally severe when they are
                # not comparable at all. Fixed rescale keeps true severity
                # visible and comparable across cases.
                reg_crop_viz = (np.clip(reg_crop, -1.0, 1.0) + 1.0) / 2.0

                # Reconstruct full-size regression volume. Fill is reg = 0, not
                # np.zeros: outside the head bbox the network produced nothing,
                # and a 0 there would store reg = -1 over 0-6% of the volume.
                reg_full = np.full(img_data.shape, REG_STORED_ZERO, dtype=np.float32)
                reg_full[bbox] = reg_crop_viz
                
                # Back to original orientation grid
                reg_full = from_ras(reg_full, inv_ornt)
                # int16 with pinned scaling (stored = raw*slope + inter): 24% smaller than
                # float32 after gzip, error <= 7.7e-6. Pin AFTER construction: Nifti1Image
                # resets scl_slope/inter. Verified by reading the file back.
                reg_q = np.round(reg_full * 65534.0 - 32767.0).astype(np.int16)
                reg_hdr = nii.header.copy(); reg_hdr.set_data_dtype(np.int16)
                reg_nii = nib.Nifti1Image(reg_q, nii.affine, reg_hdr)
                reg_nii.header.set_slope_inter(REG_INT16_SLOPE, REG_INT16_INTER)
                
                # Save alongside the segmentation file with a _reg.nii.gz suffix
                reg_out_path = out_pred_path.replace("_seg.nii.gz", "_reg.nii.gz")
                nib.save(reg_nii, reg_out_path)
                chk = nib.load(reg_out_path).dataobj
                if abs(float(chk.slope or 1.0) - REG_INT16_SLOPE) > 1e-12 or abs(float(chk.inter or 0.0) - REG_INT16_INTER) > 1e-9:
                    raise RuntimeError(f"reg map scaling not pinned on disk: {reg_out_path}")
                
            # Threshold regression output and overwrite segmentation map.
            #
            # TWO fixes from an earlier version of this block:
            # 1. Scale: the regression target used to be img_path - img_clean
            #    in RAW, pre-normalization GMM units -- a genuinely
            #    pathological voxel's target was realistically 5 to 100+,
            #    not order-1, so a threshold of 0.5 was 10-200x smaller than
            #    real signal, and ordinary prediction noise in healthy
            #    tissue cleared it, producing widespread false-positive
            #    over-segmentation. As of this retrain, the target is
            #    instead tanh((img_path - img_clean) / (p99 - p1)), where
            #    (p99 - p1) is the SAME percentile range
            #    normalize_like_training already computes for input
            #    normalization (see the training loop's own regression
            #    target construction for why raw units and dividing by
            #    img_clean's scale both failed at inference specifically,
            #    and why tanh matches the scale the regression TARGET
            #    is built in). The head has no output activation, but it is
            #    trained against that target, so its raw output already sits
            #    in this scale -- no rescaling here, only the threshold
            #    below.
            # 2. Sign: thresholding reg_crop > threshold only catches
            #    brighter-than-clean deviations. Darker pathology (chronic
            #    infarct, atrophy, WMH on T1) has a NEGATIVE regression
            #    value and would never be caught at all. Needs abs(),
            #    matching the signed nature of the target.
            #
            # REGRESSION_LESION_THRESHOLD is a reasonable STARTING POINT,
            # not an empirically calibrated value. Same numeric value as
            # before the tanh change: tanh(x) is close to linear for small
            # x, and the smallest meaningful signal (k_rose_min=1.0 *
            # sigma_min=5, divided by a typical (p99-p1) of ~150-200) is
            # ~0.025-0.033 either way, so tanh barely moves it. Ideally
            # tuned against the validation set's own regression output
            # distribution for known-healthy vs known-pathological voxels
            # before trusting Dice scores that depend on it.
            healthy_crop = pred_crop.copy()   # the seg head's own output, before any lesion overwrite
            if reg is not None:
                # reg_crop already computed above (same reg tensor,
                # unmutated in between) -- was recomputed here identically
                # a second time, pure redundant .cpu().numpy() work.
                # Only over brain: a voxel the segmentation head calls air,
                # skull or scalp cannot become Lesion.
                cerebral = cerebral_mask(pred_crop)
                pred_crop[(np.abs(reg_crop) > REGRESSION_LESION_THRESHOLD) & cerebral] = \
                    LESION_COMPACT_IDX

        pred_idx = np.zeros(img_data.shape, dtype=np.int32)
        pred_idx[bbox] = pred_crop

        if mask:
            # Only bites where the background is exact zero. Datasets that
            # keep a noisy background are handled by the cerebral restriction
            # above instead.
            pred_idx[img_data <= 0] = 0

        # Back onto the original grid so the output stays comparable with
        # ground truth stored in the source orientation.
        pred_idx = from_ras(pred_idx, inv_ornt)
        # _seg.nii.gz is the seg head's own (healthy) segmentation, untouched.
        # The |reg| overwrite is NOT saved -- lesion_mask_from() rebuilds it
        # from _reg + _seg, which is what _lesion.nii.gz held (33 GB across
        # the derivatives). Runs before 2026-09-15 stored it inside _seg.
        # compact indices < 56: uint8 halves the label maps on disk
        seg_hdr = nii.header.copy(); seg_hdr.set_data_dtype(np.uint8); seg_hdr.set_slope_inter(1.0, 0.0)
        if reg is not None:
            healthy_idx = np.zeros(img_data.shape, dtype=np.int32)
            healthy_idx[bbox] = healthy_crop
            if mask:
                healthy_idx[img_data <= 0] = 0
            nib.save(nib.Nifti1Image(from_ras(healthy_idx, inv_ornt).astype(np.uint8), nii.affine, seg_hdr), out_pred_path)
        else:
            nib.save(nib.Nifti1Image(pred_idx.astype(np.uint8), nii.affine, seg_hdr), out_pred_path)

    for flag, deriv in (("strip_mask", strip_mask), ("norm_mask", norm_mask)):
        if deriv:
            c = mask_counts[flag]
            print(f"[{flag}] {deriv}: {c.get('direct', 0)} direct, {c.get('fallback', 0)} T1w-named fallback, "
                  f"{c.get('missing', 0)} MISSING (unmasked) of {sum(c.values())} cases run")

    if diagnose_edges:
        if not edge_diag_results:
            print("\n[edge-diag] No cases produced a result (no regression head, or no files found).")
        else:
            ex_rs = np.array([r["extracerebral"]["r"] for r in edge_diag_results])
            br_rs = np.array([r["brain"]["r"] for r in edge_diag_results])
            ex_valid = ex_rs[~np.isnan(ex_rs)]
            br_valid = br_rs[~np.isnan(br_rs)]
            print("\n" + "=" * 70)
            print(f"EDGE-CORRELATION SUMMARY ({dataset}, {len(edge_diag_results)} cases)")
            print("  |reg_out| vs local gradient magnitude, split by PREDICTED tissue type")

            def _summarize(name, valid, total_n):
                skipped = total_n - len(valid)
                if len(valid) == 0:
                    print(f"  {name}: no cases had enough predicted voxels to compute a "
                          f"correlation ({skipped} skipped)")
                else:
                    print(f"  {name}: mean r={valid.mean():+.3f}  median r={np.median(valid):+.3f}  "
                          f"(n_cases={len(valid)}, {skipped} skipped: too few voxels)")

            _summarize("extracerebral", ex_valid, len(ex_rs))
            _summarize("brain tissue ", br_valid, len(br_rs))
            print("=" * 70)
            if len(ex_valid) > 0:
                print("For reference, diagnose_edge_shortcut.py's synthetic in-distribution baseline")
                print("(Test 1, on the generator's own held-out validation set) was r=+0.220. If")
                print("extracerebral here comes back well above that baseline, that is real-data")
                print("evidence the sim-to-real gap on extracerebral tissue specifically (killed 50%")
                print("of the time in training, random intensity when present) is a material,")
                print("additional source of false positives beyond what synthetic validation alone")
                print("would predict -- not a general flaw in the regression mechanism, which the")
                print("earlier partial-correlation result suggested is basically sound.")
            else:
                print("No case had enough predicted-extracerebral voxels to compute this at all --")
                print("worth checking separately whether the segmentation head predicts that class")
                print("on real data in the first place, before concluding anything about the")
                print("regression head's behavior there.")

        if not intensity_diag_results:
            print("\n[intensity-diag] No cases produced a result.")
        else:
            pm = np.array([r["partial_membership_given_intensity"] for r in intensity_diag_results])
            pi = np.array([r["partial_intensity_given_membership"] for r in intensity_diag_results])
            rm = np.array([r["r_reg_membership"] for r in intensity_diag_results])
            ri = np.array([r["r_reg_intensity"] for r in intensity_diag_results])
            pm_valid, pi_valid = pm[~np.isnan(pm)], pi[~np.isnan(pi)]
            print("\n" + "=" * 70)
            print(f"INTENSITY-VS-EXTRACEREBRAL DISENTANGLING ({dataset}, {len(intensity_diag_results)} cases)")
            print("  Skull/marrow being bright on T1 and skull being extracerebral are two")
            print("  different facts that happen to co-occur -- this separates which one is")
            print("  actually driving |reg_out|, the same way edge-riding was separated from")
            print("  genuine pathology signal earlier.")
            print()
            if len(pm_valid) > 0:
                print(f"  raw corr(reg_out, membership)  mean={np.nanmean(rm):+.3f}")
                print(f"  raw corr(reg_out, intensity)   mean={np.nanmean(ri):+.3f}")
                print(f"  PARTIAL corr(reg_out, membership | intensity) mean={pm_valid.mean():+.3f}  "
                      f"median={np.median(pm_valid):+.3f}")
                print(f"  PARTIAL corr(reg_out, intensity  | membership) mean={pi_valid.mean():+.3f}  "
                      f"median={np.median(pi_valid):+.3f}")
                print("=" * 70)
                print("If the membership partial correlation stayed close to its raw value while the")
                print("intensity partial correlation collapsed toward zero, extracerebral TISSUE TYPE")
                print("is the real driver (points toward fixing the generator's extracerebral")
                print("synthesis specifically). If the reverse -- intensity partial correlation")
                print("survives, membership collapses -- raw brightness is the real driver")
                print("regardless of tissue type (points toward a broader fix: how the GMM handles")
                print("extreme intensities in general, not an extracerebral-specific change).")
                print("Verified on synthetic data with a known answer: a pure intensity-driven")
                print("signal showed raw corr(reg,membership)=0.591 collapsing to a partial of 0.002;")
                print("a pure membership-driven signal showed the exact reverse pattern.")
            else:
                print("=" * 70)
                print("No case had enough of both groups represented to compute this.")

    return edge_diag_results, intensity_diag_results


def sweep_regression_threshold(out_dir, data_dir, dataset, modality, wanted, thresholds=None):
    """Sweeps candidate |regression output| thresholds against REAL ground
    truth, reusing already-saved _reg.nii.gz files rather than re-running
    inference. REGRESSION_LESION_THRESHOLD=0.05 (in run_inference) was
    always flagged in its own comment as a reasoned-but-uncalibrated
    starting point, never actually validated against real tumor
    annotations -- this answers "is 0.05 right" with data instead of
    another guess.

    _reg.nii.gz stores (clip(reg_raw, -1, 1) + 1) / 2 for visualization --
    a known, linear, invertible rescale. The head has NO tanh; it is only
    trained against a tanh target, so (-1, 1) holds in practice but is not
    guaranteed. Measured 2026-09-15: zero clipped voxels over 24 subjects
    across 6 checkpoints, max |reg_raw| 0.918 -- a no-op so far, not by
    construction. Recovered here via reg_raw = reg_viz*2 - 1,
    rather than needing a fresh, expensive inference pass over the whole
    dataset again just to test a different threshold value.

    Near-zero Dice with the current threshold doesn't distinguish "the
    threshold is wrong" from "the model itself isn't localizing pathology
    well yet at this checkpoint" -- both produce the same symptom. If
    every threshold in this sweep scores badly, that points at the model/
    checkpoint rather than the threshold value; if some other threshold
    scores meaningfully better than 0.05, that's a real, fixable
    threshold problem.
    """
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]

    cases = list_cases(data_dir, dataset, modality=modality)
    # Loaded once per case, reused across every threshold in the sweep,
    # rather than re-reading the same files off disk once per threshold.
    loaded = []
    for sub, t1, gtp in cases:
        seg_path = os.path.join(out_dir, output_name_for(t1, data_dir, dataset, modality))
        reg_path = seg_path.replace("_seg.nii.gz", "_reg.nii.gz")
        if not os.path.isfile(reg_path):
            continue
        reg_viz = nib.load(reg_path).get_fdata().astype(np.float32)
        reg_raw = reg_viz * 2.0 - 1.0
        gt_img = nib.load(gtp).get_fdata().astype(np.int32)
        if reg_raw.shape != gt_img.shape:
            continue
        gt_mask = np.isin(gt_img, wanted)
        loaded.append((sub, reg_raw, gt_mask))

    print(f"\n--- Threshold sweep: {len(loaded)} cases with both a saved regression map and ground truth ---")
    if not loaded:
        print("No cases had both -- run_inference needs to have been run with a checkpoint that "
              "has a regression head (saves _reg.nii.gz) for this dataset first.")
        return None

    results = {}
    for thr in thresholds:
        scores = [d for d in (dice(np.abs(reg_raw) > thr, gt_mask)
                              for _, reg_raw, gt_mask in loaded) if d is not None]
        results[thr] = (float(np.mean(scores)) if scores else float("nan"), len(scores))

    print(f"{'threshold':>10} {'mean dice':>10} {'n scored':>10}")
    for thr in thresholds:
        mean_d, n = results[thr]
        print(f"{thr:>10.3f} {mean_d:>10.4f} {n:>10d}")

    scoreable = {t: v for t, v in results.items() if not np.isnan(v[0])}
    if not scoreable:
        print("\nNo threshold produced any scoreable case (every case had empty pred AND empty "
              "ground truth at every threshold tried) -- can't identify a best value from this.")
        return results

    best_thr = max(scoreable, key=lambda t: scoreable[t][0])
    print(f"\nBest threshold in this sweep: {best_thr} (mean dice {scoreable[best_thr][0]:.4f})")
    if scoreable[best_thr][0] < 0.05:
        print("Even the best threshold in this sweep scores very poorly -- this points at the "
              "model/checkpoint itself (not yet localizing pathology well), not the threshold "
              "value. Re-running this sweep against a later, more-trained checkpoint is more "
              "informative right now than adjusting the threshold further.")
    elif best_thr != 0.05:
        print(f"Does NOT match run_inference's current REGRESSION_LESION_THRESHOLD=0.05 -- "
              f"worth updating that value to {best_thr} based on this result.")
    else:
        print("Matches run_inference's current REGRESSION_LESION_THRESHOLD=0.05.")
    return results


def calculate_dice(seg1_path, seg2_path):
    """Per-label Dice between two segmentations.

    A single foreground-vs-foreground Dice would be close to meaningless
    here: with --mask, both segmentations share exactly the same nonzero
    extent, so that term is identical by construction and every pair scores
    ~1.0. Without masking, both models would label the entire skull-stripped
    void as *something*, and that shared region would dominate the
    intersection instead of real anatomy.

    Per-label Dice measures what's actually of interest: whether the two
    models assign the same structure to the same voxels. Labels absent from
    both segmentations are skipped rather than scored 1.0, so they don't
    inflate the mean.

    Returns (mean_dice_over_present_labels, lesion_dice_or_None,
             per_label_dict).
    """
    img1 = nib.load(seg1_path).get_fdata().astype(np.int32)
    img2 = nib.load(seg2_path).get_fdata().astype(np.int32)

    labels = (set(np.unique(img1).tolist()) | set(np.unique(img2).tolist())) - {0}
    per_label = {}
    for L in sorted(labels):
        a = (img1 == L)
        b = (img2 == L)
        denom = a.sum() + b.sum()
        if denom == 0:
            continue  # absent from both -- skip, don't score as perfect
        per_label[int(L)] = 2.0 * np.logical_and(a, b).sum() / denom

    mean_dice = float(np.mean(list(per_label.values()))) if per_label else 1.0
    lesion_dice = per_label.get(LESION_COMPACT_IDX)
    return mean_dice, lesion_dice, per_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset folder (e.g., ATLAS or BraTS2023-GLI)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run inference even if the output segmentation already exists. "
                             "Needed after a normalization or checkpoint change, since cached "
                             "outputs from a previous run are otherwise silently reused.")
    parser.add_argument("--step", type=int, default=200000,
                        help="Training step of the checkpoint to evaluate (e.g. 200000, 270000). "
                             "Selects synthseg_<step>.pth for both models (unless overridden).")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Explicit path to a checkpoint to evaluate. If provided, this overrides "
                             "the default pathology checkpoint path, allowing you to run or compare "
                             "models at different step counts.")
    parser.add_argument("--single_model", action="store_true",
                        help="Only run inference on a single model (the pathology one, or the one "
                             "specified by --checkpoint_path). Skips the baseline model and Dice calculation.")
    parser.add_argument("--target-mm", type=float, default=1.0,
                        help="Voxel spacing (mm, isotropic) the input is resampled to before "
                             "segmentation, matching the training data's spacing. This is what "
                             "actually has to match between train and inference -- the network "
                             "is fully convolutional, so array size does not.")
    parser.add_argument("--no-crop", dest="crop", action="store_false",
                        help="Disable cropping to the head bounding box. Cropping is on by "
                             "default because InstanceNorm3d derives its statistics from the "
                             "input at inference time, so a volume that is mostly air normalizes "
                             "very differently from one that is mostly tissue.")
    parser.add_argument("--modality", type=str, default="3DT1",
                        help="WMH only: which volume to segment -- 3DT1 (native isotropic T1, "
                             "the default), T1 (same T1 in FLAIR space, 3mm slices), or FLAIR "
                             "(the sequence the ground truth was drawn on). The network has only "
                             "ever seen T1-like GMM contrast, so FLAIR is out of distribution in "
                             "a different way rather than necessarily better -- but WMH lesions "
                             "are hyperintense on FLAIR and near-invisible on T1, so the "
                             "comparison separates a contrast problem from everything else.")
    parser.add_argument("--mask", action="store_true",
                        help="Force predictions to background wherever the input image is zero. "
                             "Use on skull-stripped data (BraTS, ATLAS): the network was trained "
                             "on intact heads and emits arbitrary labels into the stripped void.")
    parser.add_argument("--diagnose-edges", action="store_true",
                        help="Correlate |regression output| with local image gradient magnitude, "
                             "split by predicted extracerebral vs. brain tissue -- tests whether "
                             "the sim-to-real gap on extracerebral tissue (killed 50%% of the time "
                             "in training, random intensity when present) produces a materially "
                             "worse correlation on real scans than diagnose_edge_shortcut.py's "
                             "in-distribution synthetic baseline. Runs on every case even if its "
                             "output already exists, without needing --overwrite too, since "
                             "computing this needs a fresh inference pass regardless of whether "
                             "the output file gets re-saved.")
    parser.add_argument("--sweep-threshold", action="store_true",
                        help="--single_model only: sweep candidate |regression output| thresholds "
                             "against real ground truth, reusing already-saved _reg.nii.gz files "
                             "rather than re-running inference. run_inference's own "
                             "REGRESSION_LESION_THRESHOLD=0.05 was always an uncalibrated starting "
                             "point -- this checks it against data. Requires inference to have "
                             "already been run for this dataset/checkpoint (needs the saved "
                             "_reg.nii.gz files to exist already).")
    parser.add_argument("--norm_mask", default=None, metavar="DERIV",
                        help="Diagnostic: take the [p1,p99] normalization percentiles inside this mask "
                             "derivative (e.g. synthstrip_b0) instead of the whole >0 foreground.")
    parser.add_argument("--out_suffix", default="",
                        help="Appended to the derivative directory name (keeps diagnostic runs apart).")
    parser.add_argument("--tissue_pool", default=None, choices=("own", "lr", "mirror", "coarse", "coarse_th"),
                        help="Diagnostic: override the pools of the tissue-relative channel at inference (default: the checkpoint's own). "
                             "Use with --out_suffix so the maps do not overwrite the trained configuration's.")
    parser.add_argument("--zero_refs", action="store_true",
                        help="Diagnostic: zero the head's reference channels (tissue_refs) at inference, own-class channel kept. "
                             "Use with --out_suffix.")
    parser.add_argument("--bn_mode", default="running", choices=("running", "batch"),
                        help="batch: BatchNorm uses the statistics of the image being inferred, as "
                             "every training forward did (the running buffers never entered one). "
                             "Appends _bnbatch to the derivative name.")
    parser.add_argument("--strip_mask", default=None, metavar="DERIV",
                        help="Zero the image outside this mask derivative (e.g. synthstrip_b0) before "
                             "the unchanged training normalization (percentiles over img > 0). "
                             "Appends _strip to the derivative name. Independent of --norm_mask, "
                             "which only moves the percentile frame; both may be given.")
    parser.add_argument("--target", default="whole", choices=sorted(TARGET_LABELS),
                        help="--single_model only: which BraTS sub-compartments count as ground "
                             "truth (whole=1,2,3 / core=1,3 / enhancing=3). Ignored for ATLAS/WMH, "
                             "which have a single ground-truth label regardless of this flag.")
    parser.add_argument("--save_scores", default=None, metavar="PATH",
                        help="--single_model only: append every (dataset, model, subject, dice) "
                             "row to this CSV -- creates it with a header if it doesn't exist, "
                             "appends otherwise, so BraTS/ATLAS/WMH runs accumulate into one "
                             "file. Pair with show_top5.py to summarize it.")
    parser.add_argument("--subjects", nargs="+", default=None, metavar="SUB",
                        help="Only infer/score these subjects, e.g. sub-536 or, for WMH, sub-15 / Utrecht_sub-15 "
                             "(a bare sub-N matches that number at every site). Output names and directories are "
                             "unchanged, so a two-case run drops into the same derivative directory as a full one.")
    args = parser.parse_args()
    SUBJECT_FILTER = set(args.subjects) if args.subjects else None
    if SUBJECT_FILTER:
        print(f"[subjects] restricted to {sorted(SUBJECT_FILTER)}")

    DATA_DIR = os.environ.get('PR_EVAL_DATA_ROOT', '/path/to/eval_data_root') + f"/{args.dataset}"
    SPACE_DIR = os.environ.get('PR_DERIV_ROOT', '/path/to/deriv_root') + f"/{args.dataset}"
    step = args.step
    tag = f"{step // 1000}k"
    # automatic tags after --out_suffix: _normmask_strip_bnbatch with all three on
    out_suffix = (args.out_suffix + ("_strip" if args.strip_mask else "")
                  + ("_bnbatch" if args.bn_mode == "batch" else ""))
    print(f"[bn_mode] {args.bn_mode}" + (f"   [strip_mask] {args.strip_mask}" if args.strip_mask else ""))

    # Modality suffix on the output directory too, so a FLAIR run and a 3DT1
    # run of the same checkpoint stay separate.
    mod_tag = f"_{args.modality}" if args.dataset.startswith("WMH") else ""
    BASE_OUT = os.path.join(SPACE_DIR, "derivatives", f"synthseg_baseline_{tag}{mod_tag}{out_suffix}")
    CKPT_BASE = os.environ.get('PR_BASELINE_CKPT_DIR', '/path/to/baseline_ckpt_dir') + f"/synthseg_{step}.pth"

    # Handle the custom checkpoint and output path dynamically
    if args.checkpoint_path:
        CKPT_PATH = args.checkpoint_path
        ckpt_name = os.path.basename(args.checkpoint_path).replace('.pth', '')
        
        run_name = "custom"
        if "Pathology-Randomizer/" in CKPT_PATH:
            # Extract e.g. "healthy_98ea36c" from ".../Pathology-Randomizer/healthy_98ea36c_out/..."
            sub_path = CKPT_PATH.split("Pathology-Randomizer/")[1]
            run_folder = sub_path.split("/")[0]
            if run_folder.endswith("_out"):
                run_name = run_folder[:-4]  # Strips off '_out'
            elif run_folder.endswith("out"):
                run_name = run_folder[:-3].rstrip("_")
                
        PATH_OUT = os.path.join(SPACE_DIR, "derivatives", f"synthseg_{run_name}_{ckpt_name}{mod_tag}{out_suffix}")
        model_label = f"{run_name}_{ckpt_name}"
    else:
        CKPT_PATH = os.environ.get('PR_OUT_DIR', '/path/to/out_dir') + f"/pathology_a1b8b88_healthy_out/log_20260819_112440/checkpoints/synthseg_{step}.pth"
        PATH_OUT = os.path.join(SPACE_DIR, "derivatives", f"synthseg_pathology_{tag}{mod_tag}{out_suffix}")
        model_label = f"pathology_{tag}"

    # Build a list of models to run based on the --single_model flag. Label
    # derived from the checkpoint path (not a fixed "target" string) so a
    # --save_scores CSV accumulated across several checkpoints stays
    # distinguishable by which one produced which row.
    models_to_run = [(model_label, CKPT_PATH, PATH_OUT)]
    if not args.single_model:
        models_to_run.insert(0, ("baseline", CKPT_BASE, BASE_OUT))

    # Fail fast on a missing checkpoint rather than dying inside torch.load
    # after the dataset scan has already run.
    for label, ckpt, _ in models_to_run:
        if not os.path.isfile(ckpt):
            ckpt_dir = os.path.dirname(ckpt)
            available = sorted(
                f for f in (os.listdir(ckpt_dir) if os.path.isdir(ckpt_dir) else [])
                if f.endswith(".pth")
            )
            raise SystemExit(
                f"ERROR: {label} checkpoint not found:\n  {ckpt}\n"
                + (f"\nAvailable in {ckpt_dir}:\n  " + "\n  ".join(available)
                   if available else f"\n(no .pth files in {ckpt_dir})")
            )

    # Fail fast rather than running both models over an empty file list and
    # printing an empty results table, which reads like "the models agreed on
    # nothing" instead of "the path was wrong".
    if not os.path.isdir(DATA_DIR):
        parent = os.path.dirname(DATA_DIR)
        available = sorted(os.listdir(parent)) if os.path.isdir(parent) else []
        raise SystemExit(
            f"ERROR: dataset directory does not exist:\n  {DATA_DIR}\n"
            + (f"\nAvailable datasets in {parent}:\n  " + "\n  ".join(available)
               if available else "")
        )

    probe = find_t1_files(DATA_DIR, args.dataset, args.modality)
    if not probe:
        raise SystemExit(
            f"ERROR: {DATA_DIR} exists but contains no T1 volumes matching the "
            f"expected layout for dataset '{args.dataset}'.\n"
            f"See find_t1_files() -- WMH* looks for 3DT1.nii.gz at any depth, "
            f"everything else looks for *_T1w.nii.gz under an anat/ directory."
        )
    print(f"Found {len(probe)} T1 volumes under {DATA_DIR}")

    print(f"--- Processing Dataset: {args.dataset} ---")
    for label, ckpt, out_dir in models_to_run:
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=> Running {label} model: {ckpt}")
        run_inference(ckpt, DATA_DIR, out_dir, args.dataset, overwrite=args.overwrite,
                      mask=args.mask, crop=args.crop, target_mm=args.target_mm,
                      modality=args.modality, diagnose_edges=args.diagnose_edges,
                      norm_mask=args.norm_mask, bn_mode=args.bn_mode, strip_mask=args.strip_mask, tissue_pool=args.tissue_pool, zero_refs=args.zero_refs)

    if args.single_model:
        label, ckpt, out_dir = models_to_run[0]
        key = dataset_key(args.dataset)
        wanted = DATASETS[key]["labels"] or TARGET_LABELS[args.target]
        cases = list_cases(DATA_DIR, args.dataset, modality=args.modality)
        print(f"\n--- Ground-truth Dice for {args.dataset} (single model: {label}) ---")
        print(f"{len(cases)} cases with ground truth; target labels {wanted}\n")

        csv_writer = None
        csv_file = None
        if args.save_scores:
            is_new = not os.path.isfile(args.save_scores)
            csv_file = open(args.save_scores, "a", newline="")
            csv_writer = csv.writer(csv_file)
            if is_new:
                csv_writer.writerow(["dataset", "model", "subject", "dice", "pred_vox", "gt_vox"])

        results = []
        for sub, t1, gtp in cases:
            pred_path = os.path.join(out_dir, output_name_for(t1, DATA_DIR, args.dataset, args.modality))
            if not os.path.isfile(pred_path):
                print(f"  {sub}: SKIPPED, no prediction at {pred_path}")
                continue
            # Both files are already on the original (scanner) grid --
            # run_inference wrote the prediction there via from_ras() before
            # saving, ground truth ships in that same native orientation --
            # so no RAS conversion here, just the same grid-match guard
            # eval_lesion_dice.py uses.
            lesion_path = pred_path.replace("_seg.nii.gz", "_lesion.nii.gz")
            reg_path = pred_path.replace("_seg.nii.gz", "_reg.nii.gz")
            if os.path.isfile(lesion_path):         # written before 2026-09-16
                pred_mask = nib.load(lesion_path).get_fdata() > 0
            elif os.path.isfile(reg_path):
                pred_mask = lesion_mask_from(reg_path, pred_path)
            else:   # runs that stored the overwrite inside _seg
                pred_mask = nib.load(pred_path).get_fdata().astype(np.int32) == LESION_COMPACT_IDX
            gt_img = nib.load(gtp).get_fdata().astype(np.int32)
            if pred_mask.shape != gt_img.shape:
                print(f"  {sub}: SKIPPED, grid mismatch pred={pred_mask.shape} vs gt={gt_img.shape}")
                continue
            gt_mask = np.isin(gt_img, wanted)
            d = dice(pred_mask, gt_mask)
            results.append((sub, d, int(pred_mask.sum()), int(gt_mask.sum())))

            if csv_writer:
                # d can be None (both masks empty) -- written as an empty
                # field rather than skipped, so row count still matches
                # cases attempted.
                csv_writer.writerow([args.dataset, label, sub,
                                     f"{d:.6f}" if d is not None else "",
                                     int(pred_mask.sum()), int(gt_mask.sum())])
                csv_file.flush()

            ds = "  n/a" if d is None else f"{d:5.3f}"
            print(f"  {sub}  dice {ds}   pred {pred_mask.sum():7d}  gt {gt_mask.sum():7d}")

        if csv_file:
            csv_file.close()
            print(f"\nscores appended to {args.save_scores}")

        scored = np.array([r[1] for r in results if r[1] is not None])
        print("\n" + "=" * 60)
        if len(scored):
            print(f"{label:<14} dice {scored.mean():.3f} \u00b1 {scored.std():.3f}  "
                  f"[{scored.min():.3f}, {scored.max():.3f}]  n={len(scored)}")
        else:
            print(f"{label:<14} no scorable cases")
        print("=" * 60)

        if args.sweep_threshold:
            sweep_regression_threshold(out_dir, DATA_DIR, args.dataset, args.modality, wanted)
    else:
        print(f"\n--- Calculating Dice Scores for {args.dataset} ---")
        results = []
        label_accum = {}
        for filename in sorted(os.listdir(BASE_OUT)):
            if filename.endswith(".nii.gz"):
                p1 = os.path.join(BASE_OUT, filename)
                p2 = os.path.join(PATH_OUT, filename)
                if os.path.exists(p2):
                    mean_d, lesion_d, per_label = calculate_dice(p1, p2)
                    results.append((filename, mean_d, lesion_d))
                    for L, v in per_label.items():
                        label_accum.setdefault(L, []).append(v)

        sorted_results = sorted(results, key=lambda x: x[1])
        print("\n" + "=" * 78)
        print(f"{args.dataset + ' Subject File':<50} | {'Mean Dice':<10} | {'Lesion':<8}")
        print("=" * 78)
        for acquisition, mean_d, lesion_d in sorted_results:
            lesion_str = f"{lesion_d:.4f}" if lesion_d is not None else "   --   "
            print(f"{acquisition:<50} | {mean_d:.4f}     | {lesion_str}")
        print("=" * 78)
        print(f"Total evaluated pairs: {len(sorted_results)}")

        if sorted_results:
            overall = np.mean([r[1] for r in sorted_results])
            lesion_vals = [r[2] for r in sorted_results if r[2] is not None]
            print(f"Overall mean Dice across subjects: {overall:.4f}")
            if lesion_vals:
                print(f"Lesion (compact idx {LESION_COMPACT_IDX}) mean Dice: {np.mean(lesion_vals):.4f} "
                      f"(present in {len(lesion_vals)}/{len(sorted_results)} subjects)")
            else:
                print(f"Lesion (compact idx {LESION_COMPACT_IDX}) absent from all subjects in both models")

        if label_accum:
            print("\n" + "-" * 78)
            print("Per-label agreement, lowest first (where the two models disagree most)")
            print("-" * 78)
            by_label = sorted(((L, float(np.mean(v)), len(v)) for L, v in label_accum.items()),
                              key=lambda x: x[1])
            for L, v, n in by_label[:15]:
                print(f"  label {L:<3} | dice {v:.4f} | present in {n} subjects")
        print()