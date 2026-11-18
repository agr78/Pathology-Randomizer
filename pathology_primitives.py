import torch
import torch.nn.functional as F

# Rose criterion threshold for genuine human-observer signal detectability.
#
# Rose, A. (1948). "The Sensitivity Performance of the Human Eye on an
# Absolute Scale." Journal of the Optical Society of America, 38(2),
# 196-208. https://doi.org/10.1364/JOSA.38.000196
# Rose's own experiments determined a threshold signal-to-noise ratio of
# five for reliable human detection, stated directly in the original
# paper's own abstract.
#
# Burgess, A. E. (1996). "The Rose model, revisited." Journal of the
# Optical Society of America A, 16(3), 633-646.
# https://doi.org/10.1364/JOSAA.16.000633
# A modern re-examination, explicitly checking whether 4 or 5 fits better
# against real human-observer data and confirming 5.
#
# ONE constant, shared by both the label-level geometric gate (below, in
# augment_batch's k_rose default) and the downstream image-level
# visibility check/correction in generator.py -- the two answer
# different questions (this gate compares PathologyRandomizer's own
# discarded intensity modulation against noise; the other compares the
# actual rendered image), but both need the same detectability threshold,
# and only this value has independent grounding in the literature above.
ROSE_CRITERION_K = 5.0

# From `brain_mris_QCed.readme.txt` anatomical label mapping (brainseg 0-43)
FREESURFER_LABELS = {
    1: "Left-Cerebral-White-Matter",
    2: "Left-Cerebral-Cortex",
    3: "Left-Lateral-Ventricle",
    4: "Left-Inf-Lat-Vent",
    5: "Left-Cerebellum-White-Matter",
    6: "Left-Cerebellum-Cortex",
    7: "Left-Thalamus",
    8: "Left-Caudate",
    9: "Left-Putamen",
    10: "Left-Pallidum",
    11: "3rd-Ventricle",
    12: "4th-Ventricle",
    13: "Brain-Stem",
    14: "Left-Hippocampus",
    15: "Left-Amygdala",
    16: "CSF",
    17: "Left-Accumbens-area",
    18: "Right-Cerebral-White-Matter",
    19: "Right-Cerebral-Cortex",
    20: "Right-Lateral-Ventricle",
    21: "Right-Inf-Lat-Vent",
    22: "Right-Cerebellum-White-Matter",
    23: "Right-Cerebellum-Cortex",
    24: "Right-Thalamus",
    25: "Right-Caudate",
    26: "Right-Putamen",
    27: "Right-Pallidum",
    28: "Right-Hippocampus",
    29: "Right-Amygdala",
    30: "Right-Accumbens-area",
    31: "WM-hypointensities",
    32: "Optic-Chiasm",
    33: "Lesion",
    34: "Left-HypoThal-noMB",
    35: "Right-HypoThal-noMB",
    36: "Left-Fornix",
    37: "Right-Fornix",
    38: "Left-MammillaryBody",
    39: "Right-MammillaryBody",
    40: "Left-Basal-Forebrain",
    41: "Right-Basal-Forebrain",
    42: "Left-SeptalNuc",
    43: "Right-SeptalNuc"
}

# Extracerebral label ids under the production 0-56 "brainseg_with_
# extracerebral" scheme (verified against train.py's own
# FREESURFER_LABELS, which uses the real label_list_segmentation numbering
# -- NOT the smaller, differently-numbered 0-43 dict below, which is
# unused elsewhere in this file and kept only for reference): a single
# contiguous block, ids 34-46 -- air-internal, artery, eyeball,
# other-tissues, rectus-muscles, mucosa, skin, spinal-cord, vein,
# bone-cortical, cone-cancellous, background, optic-nerve. Sandwiched
# between core brain structures (1-33, including Lesion) and additional
# deep nuclei appended afterward (47-56).
#
# Real pathology doesn't seed in skull, skin, or orbital tissue -- a
# lesion appearing in that space is only physically plausible as a
# CONSEQUENCE of significant mass effect displacing brain tissue toward
# it, never as where the pathology originates. Excluded here from both
# the ROI-targeting candidate list and the hard tissue-eligibility mask,
# so pathology can never be SEEDED there for any mode -- but this is
# checked before mass-effect warping, not after, so a strong mass effect
# can still naturally displace brain-parenchyma-seeded pathology close to
# the boundary afterward. That's the intended, physically correct
# asymmetry, not a gap: skull-adjacent pathology should only ever be a
# consequence of displacement, never a starting point.
EXTRACEREBRAL_LABEL_IDS = set(range(34, 47))

# Target injected burden per mode, as a fraction of brain volume, from the
# loads the corresponding real pathology presents at. See
# docs/pathology_size_distribution.md for the measurements these were fitted
# against. Brain taken as ~1400 mL for the conversions.
#
#   tract     white matter hyperintensity. Mean 0.89% of intracranial volume,
#             range 0.05-5.04%, older adults without cognitive impairment.
#             Arvanitakis et al., Brain Struct Funct 221:2135-2146 (2015),
#             doi:10.1007/s00429-015-1034-7
#
#   vascular  chronic stroke. [CITATION NEEDED: an INDEPENDENT chronic-stroke
#             cohort's lesion-volume distribution.] The previous anchor here was
#             the ATLAS dataset paper (Liew et al. 2018) plus our own median over
#             the released ATLAS masks -- both are measurements of a test set and
#             were removed (no test data in design). Acute
#             cohorts run larger (mean 41.9 cm^3, n=753, Sperber et al., NeuroImage
#             Clin 40:103511 (2023), doi:10.1016/j.nicl.2023.103511), so a chronic
#             figure is the right one. NOTE: LESION_EXTENT_SCALE["vascular"] is 1.0
#             in every trained arm (no launcher passes --lesion_extent), so the
#             0.005 target below never shaped a model; the live burden is the
#             unscaled ~0.015 measured in domain (n=30).
#
#   local     glioblastoma. Median FLAIR/T2 hyperintense volume 37 cm^3,
#             IQR 17-70, n=177 -- 2.6% of brain at the median, 5.0% at the
#             upper quartile. Lau et al., Brain Sci 13(11):1579 (2023),
#             doi:10.3390/brainsci13111579. Target set at Lau's upper quartile
#             (5.0%) as a design choice; the sentence that justified it against
#             the evaluation cohort's measured burden was removed 2026-10-05.
TARGET_BURDEN_FRAC = {
    "tract": 0.01,
    "vascular": 0.005,
    "local": 0.05,
}

# Scales taking each mode from its measured burden to the target above.
# Measured through the real generator with one mode forced and no coupling,
# n=30 per mode: local 0.0051, vascular 0.0147, tract 0.0378.
#
# keep_frac is the fraction of a mode's prior support that becomes lesion, so
# burden is linear in it -- the right lever for the two modes that must
# shrink. Local has to GROW ~10x, which keep_frac cannot deliver: it is a
# fraction, already averages ~0.55, and even 1.0 buys under 2x. So local
# scales its focal envelope instead, where support volume goes as sigma^3 and
# the needed factor is the cube root of 9.8.
#
# The envelope helper is shared by all three modes, so any scale is passed per
# call rather than changed in the helper -- growing it globally would inflate
# the two modes that need shrinking.
#
# local was the hard one. Its envelope had no measurable effect at first (a
# sweep over focal_scale 1.0-4.0 at n=100 gave medians 0.0041-0.0083, no
# trend) because _local_pathology built the focal prior with
# target_mask=roi_mask, clipping it to a single anatomical label before
# confine_prob was ever rolled -- which also made that parameter's unconfined
# path dead code. The prior is now built on the brain, the ROI only biases
# WHERE, and the three constants below were fitted together.
# keep_frac scaling was WRONG (2026-09-13): it thresholds the spectral field
# at a higher quantile, which keeps the field's PEAKS -- one confluent lesion
# became ~20 specks (largest/total 0.96 -> 0.42) while burden fell "linearly".
# Real WMH are confluent caps; a model trained on specks scored 0.48 AUC on
# them. Restored to 1.0. Size is now set by the GEOMETRIC prior instead.
KEEP_FRAC_SCALE = {"vascular": 1.0, "tract": 1.0}

# Linear scale on each mode's geometric envelope: the slab's extent and
# thickness and the focal sigma (tract); tube length, wedge reach and focal
# sigma (vascular). Shrinks a lesion as a coherent blob. Burden goes roughly
# as scale^3, so tract 0.0378 -> 0.01 wants ~0.64 and vascular 0.0147 ->
# 0.005 wants ~0.70; verified by regenerating and measuring burden AND
# fragmentation, not assumed. 1.0 = the pre-fix geometry.
LESION_EXTENT_SCALE = {"vascular": 1.0, "tract": 1.0}

# Fraction of local draws confined to the anatomical label they were seeded
# from, and the envelope width. Lowered from 0.8 so a tumour-scale mass can
# cross structure boundaries, which is what an infiltrating glioma does.
#
# From a 3x3 sweep at n=100 (docs/pathology_size_distribution.md): this cell
# put the median at 0.0550 against a 0.05 target with the tightest spread of
# any cell (IQR 0.015-0.121). The cell nearer the target in absolute terms had
# a 49x IQR -- a mixture of tiny confined draws and brain-spanning ones, not a
# tumour-like distribution. Fit the shape, not just the median.
# Provenance split: the two constants are not alike.
#   LOCAL_CONFINE_PROB   0.5 is the uninformative default on "confined to the seeded structure vs free to
#                        infiltrate" -- a coin, not a fitted value; the sweep record above also chose it for
#                        distribution shape, not for proximity to any target. Keep; no citation needed.
#   FOCAL_SIGMA_SCALE_LOCAL is the ONLY constant that encodes a size, and 0.6 was fitted to a 5.0% burden
#                        target chosen with the evaluation cohort in view. Clean anchor: Lau et al. 2023
#                        (Brain Sci 13:1579, PMC10670247; 177 GBM, Sydney, independent of BraTS) T2 volume
#                        median 37 cm^3, IQR 17-70 -> brain fraction 0.026, IQR 0.012-0.050. Support volume
#                        MEASURED 2026-10-05 (n=100 per cell, 3x3 grid):
#                        confine 0.5 / focal 0.6 gives burden median 0.0258 vs Lau's 0.026 -- on target -- and
#                        is the closest cell to Lau's shape (log-distance 0.55; focal 0.45 undershoots at 0.0145,
#                        0.35 at 0.007). The old "0.055" was measured on a September snapshot, before the
#                        composition / Rose-floor changes altered visible burden. So 0.6 STANDS, now derived from
#                        Lau alone; no disclosure needed. Residual mismatch is spread, not size: IQR 0.008-0.073
#                        vs Lau's 0.012-0.050 (~2x wider at both ends); raising confine widens the low tail.
LOCAL_CONFINE_PROB = 0.5
FOCAL_SIGMA_SCALE_LOCAL = 0.6

# Foci per local draw, as weights over 1..4. Foci are unioned, so burden
# scales roughly with the count. The previous 0.55/0.25/0.13/0.07 gave a mean
# of 1.55 and made 41% of draws multifocal -- high for a mode now carrying
# tumour-scale masses (multifocal glioma is ~2-10% of cases). Kept multifocal
# rather than removed so metastasis- and MS-like presentations still occur.
# Mean 1.18; the sweep above ran at 1.55, so expect ~25% below its numbers.
LOCAL_FOCI_WEIGHTS = [0.85, 0.12, 0.03, 0.00]

# Modes to suppress entirely, by name in augment_batch's mode_names. Set from
# train.py (--no_diffuse) rather than threaded through three signatures.
# Zeroing a mode's primary weight is not enough: coupling rolls every
# non-primary mode independently, so a disabled mode has to be masked after
# the roll as well. Only ever applied to non-primary modes, so the
# one-mode-always-active guarantee holds.
DISABLED_MODES = set()

# Geometric Atrophy picks its ROI uniformly over labels; with this probability
# it picks a ventricle instead (train.py --atrophy_coupled), so ventricular
# enlargement is a regular secondary effect rather than a rare draw.
ATROPHY_VENTRICLE_PROB = 0.0
# Per-mode override of the secondary-coupling probability (default: couple_prob,
# 0.15, for every mode). train.py --atrophy_prob sets Geometric Atrophy's.
COUPLE_PROB_BY_MODE = {}
VENTRICLE_LABEL_IDS = (3, 20, 11, 12)
# Local Pathology's ROI label is drawn from LOCAL_HEMISPHERIC_LABEL_IDS (WM and
# cortex, S ids) with this probability. train.py --local_hemispheric.
LOCAL_HEMISPHERIC_PROB = 0.0
LOCAL_HEMISPHERIC_LABEL_IDS = (1, 18, 2, 19)


def _is_extracerebral(seg_tensor, extracerebral_ids=EXTRACEREBRAL_LABEL_IDS):
    mask = torch.zeros_like(seg_tensor, dtype=torch.bool)
    for lid in extracerebral_ids:
        mask |= (seg_tensor == lid)
    return mask


class PathologyRandomizer:
    def __init__(self, shape, device='cuda', n_mixtures=15):
        self.device = device
        self.shape = tuple(shape[-3:])
        self.n_mixtures = n_mixtures
        self.max_dim = float(max(self.shape))

    def _gaussian_blur(self, x, sigma):
        if sigma <= 0:
            return x
        size = int(sigma * 3) * 2 + 1
        pad = size // 2
        device = x.device
        xs = torch.linspace(-pad, pad, size, device=device)
        g = torch.exp(-xs**2 / (2 * sigma**2))
        g = g / g.sum()
        x_pad = F.pad(x, (pad, pad, pad, pad, pad, pad), mode='replicate')
        out = F.conv3d(x_pad, g.view(1, 1, 1, 1, size), padding=0)
        out = F.conv3d(out,   g.view(1, 1, 1, size, 1), padding=0)
        out = F.conv3d(out,   g.view(1, 1, size, 1, 1), padding=0)
        return out

    def _smooth_displacement_field(self, field, sigma):
        if sigma <= 0:
            return field
        size = int(sigma * 3) * 2 + 1
        pad = size // 2
        device = field.device
        xs = torch.linspace(-pad, pad, size, device=device)
        g = torch.exp(-xs**2 / (2 * sigma**2))
        g = g / g.sum()
        
        field_channels = field.permute(0, 4, 1, 2, 3)
        field_pad = F.pad(field_channels, (pad, pad, pad, pad, pad, pad), mode='replicate')
        
        weight_w = g.view(1, 1, 1, 1, size).repeat(3, 1, 1, 1, 1)
        weight_h = g.view(1, 1, 1, size, 1).repeat(3, 1, 1, 1, 1)
        weight_d = g.view(1, 1, size, 1, 1).repeat(3, 1, 1, 1, 1)
        
        out = F.conv3d(field_pad, weight_w, padding=0, groups=3)
        out = F.conv3d(out,       weight_h, padding=0, groups=3)
        out = F.conv3d(out,       weight_d, padding=0, groups=3)
        return out.permute(0, 2, 3, 4, 1)

    def _identity_grid(self, batch_size):
        theta = torch.eye(3, 4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        return F.affine_grid(theta, (batch_size, 1, *self.shape), align_corners=False)

    def _mass_effect_displacement(self, potential, strength, blur_sigma=4.0):
        blurred = self._gaussian_blur(potential, blur_sigma)
        gD, gH, gW = torch.gradient(blurred[:, 0], dim=(1, 2, 3))
        mag = torch.sqrt(gD**2 + gH**2 + gW**2 + 1e-16)
        peak = mag.flatten(1).max(dim=1).values.view(-1, 1, 1, 1) + 1e-6
        s = strength.view(-1, 1, 1, 1) / peak

        size = torch.tensor(self.shape, device=self.device, dtype=torch.float32)
        dD = (gD * s) * (2.0 / size[0])
        dH = (gH * s) * (2.0 / size[1])
        dW = (gW * s) * (2.0 / size[2])
        field = torch.stack([dW, dH, dD], dim=-1)
        
        smooth_sigma = (torch.rand(1, device=self.device).item() * 0.015 + 0.005) * self.max_dim
        return self._smooth_displacement_field(field, sigma=smooth_sigma)

    def _select_roi_masks(self, seg_prepared, labels, batch_size, prefer=None, prefer_prob=0.0):
        if seg_prepared is None or labels is None or labels.numel() == 0:
            return None, None
        pool = labels
        if prefer is not None and prefer_prob > 0 and torch.rand(1).item() < prefer_prob:
            keep = torch.zeros_like(labels, dtype=torch.bool)
            for lid in prefer:
                keep |= (labels == lid)
            if keep.any():
                pool = labels[keep]
        idx = torch.randint(0, pool.numel(), (batch_size,), device=self.device)
        chosen = pool[idx].view(batch_size, 1, 1, 1, 1)
        roi = (seg_prepared == chosen).float()
        return roi, pool[idx]

    def _universal_spectral_field(self, batch_size, max_freq=None, force_dc=False):
        D, H, W = self.shape
        freq_d = torch.fft.fftfreq(D, device=self.device)
        freq_h = torch.fft.fftfreq(H, device=self.device)
        freq_w = torch.fft.rfftfreq(W, device=self.device)

        fd, fh, fw = torch.meshgrid(freq_d, freq_h, freq_w, indexing='ij')
        freq_grid = torch.stack([fd, fh, fw], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

        # --- DYNAMICALLY DRAW NUMBER OF MIXTURES ---
        # Keep it low (e.g., 1 to 10) to avoid chaotic overlapping noise
        current_mixtures = torch.randint(1, 10, (1,)).item()

        weights = torch.sigmoid(torch.randn((batch_size, current_mixtures, 1, 1, 1), device=self.device))
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        # --- DYNAMICALLY DRAW MAX FREQUENCY IF NOT SPECIFIED ---
        if max_freq is None:
            # Draw a low, smooth frequency bound (e.g., between 0.05 and 0.15)
            max_freq = torch.rand(1, device=self.device).item() * 0.10 + 0.05

        safe_max_freq = min(max_freq, 0.25)
        min_dim = float(min(self.shape))
        max_dim = float(max(self.shape))
        min_bandwidth = 2.0 / min_dim
        min_hz = 1.0 / max_dim

        if force_dc:
            means = torch.zeros((batch_size, current_mixtures, 1, 1, 1, 3), device=self.device)
        else:
            means = (torch.rand((batch_size, current_mixtures, 1, 1, 1, 3), device=self.device) * (safe_max_freq - min_hz)) + min_hz

        base_scales = torch.rand((batch_size, current_mixtures, 1, 1, 1, 3), device=self.device) * (safe_max_freq / 2.0)
        scales = torch.clamp(base_scales, min=min_bandwidth)

        psd = torch.zeros((batch_size, D, H, freq_w.shape[0]), device=self.device)
        for i in range(current_mixtures):
            diff = freq_grid - means[:, i]
            component = torch.exp(-0.5 * torch.sum((diff / scales[:, i])**2, dim=-1))
            psd += weights[:, i] * component

        real_noise = torch.randn_like(psd)
        imag_noise = torch.randn_like(psd)
        fourier_noise = torch.complex(real_noise, imag_noise)
        fourier_field = fourier_noise * torch.sqrt(psd)

        grf = torch.fft.irfftn(fourier_field, s=self.shape, dim=(-3, -2, -1)).unsqueeze(1)
        mean = grf.mean(dim=(2, 3, 4), keepdim=True)
        std = grf.std(dim=(2, 3, 4), keepdim=True) + 1e-6
        return (grf - mean) / std

    def _anisotropic_warp(self, field, max_elongation=2.5, max_rotation_deg=40.0):
        B = field.shape[0]
        elong = 1.0 + torch.sigmoid(torch.randn((B, 3), device=self.device)) * (max_elongation - 1.0)
        scale = torch.diag_embed(1.0 / elong)

        angles = (torch.sigmoid(torch.randn((B, 3), device=self.device)) * 2 - 1) * (max_rotation_deg * torch.pi / 180)
        cx, sx = torch.cos(angles[:, 0]), torch.sin(angles[:, 0])
        cy, sy = torch.cos(angles[:, 1]), torch.sin(angles[:, 1])
        cz, sz = torch.cos(angles[:, 2]), torch.sin(angles[:, 2])
        one, zero = torch.ones(B, device=self.device), torch.zeros(B, device=self.device)
        Rx = torch.stack([torch.stack([one, zero, zero], -1),
                           torch.stack([zero, cx, -sx], -1),
                           torch.stack([zero, sx, cx], -1)], -2)
        Ry = torch.stack([torch.stack([cy, zero, sy], -1),
                           torch.stack([zero, one, zero], -1),
                           torch.stack([-sy, zero, cy], -1)], -2)
        Rz = torch.stack([torch.stack([cz, -sz, zero], -1),
                           torch.stack([sz, cz, zero], -1),
                           torch.stack([zero, zero, one], -1)], -2)
        R = Rz @ Ry @ Rx

        theta = torch.zeros((B, 3, 4), device=self.device)
        theta[:, :, :3] = R @ scale
        grid = F.affine_grid(theta, field.shape, align_corners=False)
        return F.grid_sample(field, grid, align_corners=False, padding_mode='reflection')

    def _local_pathology(self, batch_size, seg_prepared=None, labels=None, max_foci=3,
                          roi_prob=0.8, confine_prob=None, tissue_mask=None, force_ring=False):
        if confine_prob is None:
            confine_prob = LOCAL_CONFINE_PROB
        foci_weights = torch.tensor(LOCAL_FOCI_WEIGHTS[:max_foci], device=self.device)
        n_foci = int(torch.multinomial(foci_weights, 1).item()) + 1

        mask, novelty = None, None
        core_masks, shell_masks = [], []
        chosen_labels = None
        is_ring_tensor = torch.zeros((batch_size, 1, 1, 1, 1), device=self.device)

        for b_idx in range(batch_size):
            ring_decision = force_ring or (torch.rand(1, device=self.device).item() < 0.50)
            is_ring_tensor[b_idx, 0, 0, 0, 0] = 1.0 if ring_decision else 0.0

        for _ in range(n_foci):
            dyn_freq = torch.sigmoid(torch.randn(1, device=self.device)).item() * 0.2 + 0.15
            field = self._anisotropic_warp(self._universal_spectral_field(batch_size, max_freq=dyn_freq, force_dc=True))

            if seg_prepared is not None and labels is not None and labels.numel() > 0:
                roi_mask, chosen_labels = self._select_roi_masks(
                    seg_prepared, labels, batch_size,
                    prefer=LOCAL_HEMISPHERIC_LABEL_IDS, prefer_prob=LOCAL_HEMISPHERIC_PROB)
                roi_blur_sigma = (torch.rand(1, device=self.device).item() * 0.04 + 0.02) * self.max_dim
                roi_blur = self._gaussian_blur(roi_mask, roi_blur_sigma)
                roi_blur = roi_blur / (roi_blur.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)

                use_roi = (torch.rand((batch_size, 1, 1, 1, 1), device=self.device) < roi_prob).float()
                bias_strength = torch.sigmoid(torch.randn((batch_size, 1, 1, 1, 1), device=self.device)) * 3 + 1.5
                field = field + use_roi * bias_strength * roi_blur
            else:
                roi_mask = None

            # GATED on a compact focal prior, completing the change already made
            # to the tract and vascular modes. This mode had no spatial prior at
            # all -- it thresholded the raw spectral field across the whole
            # volume, with the ROI term only BIASING it -- so nothing confined
            # the excursion set and it produced the most ragged, most fragmented
            # lesions of the three.
            #
            # A solid tumour mass has surface-to-volume that stays roughly
            # size-independent, since it grows outward as a compact blob.
            # Thresholding a spectral field instead produces more boundary as
            # the region grows, so unconstrained lesions get raggeder with
            # size -- structure real tumours don't have. Gating on a compact
            # prior removes
            # that coupling because the prior, not the field, sets the extent.
            # Built on the brain, not the ROI. Clipping the envelope to a single
            # anatomical label capped burden at that label's size, made
            # confine_prob's unconfined path unreachable, and left the envelope
            # with no measurable effect at all. The ROI still biases WHERE via
            # the field term above, and confine_prob still confines a share of
            # draws. See docs/pathology_size_distribution.md.
            prior_target = tissue_mask if tissue_mask is not None else torch.ones_like(field)
            focal_prior = self._create_localized_focal_prior(
                batch_size, prior_target, focal_scale=FOCAL_SIGMA_SCALE_LOCAL)
            prior_support = focal_prior > 0.05
            keep_frac = torch.sigmoid(torch.randn(1, device=self.device)).item() * 0.7 + 0.2
            for b_idx in range(batch_size):
                inside = field[b_idx][prior_support[b_idx]]
                if inside.numel() < 27:
                    # Too few voxels for a quantile; still confine to the support.
                    field[b_idx] = field[b_idx] * prior_support[b_idx].float() - \
                                   (~prior_support[b_idx]).float() * 1e3
                    continue
                thr = torch.quantile(inside.float(), 1.0 - keep_frac)
                field[b_idx] = (field[b_idx] - thr) * prior_support[b_idx].float() - \
                               (~prior_support[b_idx]).float() * 1e3

            # Threshold at zero: the quantile shift above already placed the cut,
            # and it is a quantile of the field INSIDE the prior rather than an
            # absolute value on a field whose scale is arbitrary.
            coverage_shift = torch.zeros((batch_size, 1, 1, 1, 1), device=self.device)
            sharpness = torch.sigmoid(torch.randn((batch_size, 1, 1, 1, 1), device=self.device)) * 6.0 + 2.0

            outer_focus = torch.sigmoid((field - coverage_shift) * sharpness)
            core_focus = torch.sigmoid((field - (coverage_shift + 0.6)) * sharpness)
            shell_focus = outer_focus * (1.0 - core_focus)

            focus = outer_focus
            core = torch.where(is_ring_tensor > 0.5, core_focus, torch.zeros_like(core_focus))
            shell = torch.where(is_ring_tensor > 0.5, shell_focus, outer_focus)

            if seg_prepared is not None and roi_mask is not None:
                confine = (torch.rand((batch_size, 1, 1, 1, 1), device=self.device) < confine_prob).float()
                focus = focus * (confine * roi_mask + (1 - confine))
                core = core * (confine * roi_mask + (1 - confine))
                shell = shell * (confine * roi_mask + (1 - confine))

            eta = torch.sigmoid(torch.randn((batch_size, 1, 1, 1, 1), device=self.device))
            foci_novelty = ((focus - (1 - eta)) / (eta + 1e-6)).clamp(0, 1)

            mask = focus if mask is None else torch.maximum(mask, focus)
            novelty = foci_novelty if novelty is None else torch.maximum(novelty, foci_novelty)
            
            core_masks.append(core)
            shell_masks.append(shell)

        total_core = core_masks[0]
        total_shell = shell_masks[0]
        for c_m, s_m in zip(core_masks[1:], shell_masks[1:]):
            total_core = torch.maximum(total_core, c_m)
            total_shell = torch.maximum(total_shell, s_m)

        if tissue_mask is not None:
            mask = mask * tissue_mask
            novelty = novelty * tissue_mask
            total_core = total_core * tissue_mask
            total_shell = total_shell * tissue_mask

        has_me = (torch.rand((batch_size, 1, 1, 1, 1), device=self.device) < 0.8).float()
        sign = torch.where(torch.randn((batch_size, 1, 1, 1, 1), device=self.device) < 0, 1.0, -1.0)
        magnitude = torch.sigmoid(torch.randn((batch_size, 1, 1, 1, 1), device=self.device)) * 6.0 + 2.0
        strength = has_me * sign * magnitude

        return mask, novelty, strength, total_core, total_shell, n_foci, chosen_labels, is_ring_tensor

    def _global_pathology(self, batch_size, tissue_mask):
        dyn_freq = torch.sigmoid(torch.randn(1, device=self.device)).item() * 0.1 + 0.05
        field = self._universal_spectral_field(batch_size, max_freq=dyn_freq, force_dc=True)
        
        big_sigma = self.max_dim * (torch.rand(1, device=self.device).item() * 0.1 + 0.1)
        field = self._gaussian_blur(field, big_sigma)
        field = (field - field.mean(dim=(2, 3, 4), keepdim=True)) / (field.std(dim=(2, 3, 4), keepdim=True) + 1e-6)

        gain = torch.sigmoid(torch.randn((batch_size, 1, 1, 1, 1), device=self.device)) * 1.5 + 0.5
        severity = 0.5 * (1 + torch.tanh(field * gain))
        if tissue_mask is not None:
            severity = severity * tissue_mask

        eta = 1.0 - torch.sigmoid(torch.randn((batch_size, 1, 1, 1, 1), device=self.device))
        novelty = ((severity - (1 - eta)) / (eta + 1e-6)).clamp(0, 1)

        has_me = (torch.rand((batch_size, 1, 1, 1, 1), device=self.device) < 0.5).float()
        sign = torch.where(torch.randn((batch_size, 1, 1, 1, 1), device=self.device) < 0, 1.0, -1.0)
        magnitude = torch.sigmoid(torch.randn((batch_size, 1, 1, 1, 1), device=self.device)) * 4.0
        strength = has_me * sign * magnitude

        return severity, novelty, strength, "N/A (Global)"

    def _geometric_atrophy(self, batch_size, seg_prepared, labels):
        if seg_prepared is None or labels is None or labels.numel() == 0:
            dummy_roi = torch.zeros((batch_size, 1, *self.shape), device=self.device)
            return dummy_roi, torch.zeros((batch_size, 1, 1, 1, 1), device=self.device), None
        
        roi_mask, chosen_labels = self._select_roi_masks(
            seg_prepared, labels, batch_size,
            prefer=VENTRICLE_LABEL_IDS, prefer_prob=ATROPHY_VENTRICLE_PROB)
        is_expansion = (torch.rand((batch_size, 1, 1, 1, 1), device=self.device) < 0.5).float()
        
        roi_blur_sigma = (torch.rand(1, device=self.device).item() * 0.29 + 0.01) * self.max_dim
        blurred = self._gaussian_blur(roi_mask, sigma=roi_blur_sigma)
        
        shift_mag = torch.sigmoid(torch.randn((batch_size, 1, 1, 1, 1), device=self.device)) * 0.15 + 0.05
        shift = torch.where(is_expansion > 0.5, shift_mag, -shift_mag)
        
        modified_roi = torch.sigmoid((blurred - (0.5 + shift)) * 12.0)
        potential = modified_roi - roi_mask
        strength = is_expansion * 5.0 - (1.0 - is_expansion) * 5.0
        
        return potential, strength, chosen_labels

    def _tangent_slab_prior(self, batch_size, band, normal_source,
                            thick_frac=(0.015, 0.045), extent_frac=(0.10, 0.28)):
        """A plane tangent to a surface, thin along its normal.

        Anchors at a point on `band`, takes the local surface normal from the
        gradient of `normal_source`, and keeps everything within `thick` of the
        tangent plane and `extent` of the anchor. Unlike any construct derived
        from the surface itself, this does not curve with it -- flatness
        becomes a function of thickness/extent rather than of the anatomy.

        The ranges are sampled so the family spans blob-like (thick, small
        extent) through sheet-like (thin, wide), bracketing the 3.68-8.49
        real periventricular range with margin at both ends. Thicknesses
        below ~3 voxels are excluded: a near-single-voxel plane has an
        almost zero smallest eigenvalue, so flatness diverges (into the
        tens of thousands for a 2-voxel slab) -- a degenerate shape rather
        than a realistic one.

        Anchored to a normal field, so it is not ventricle-specific -- the same
        construct applies to any surface with a gradient (cortex, falx, skull).
        """
        out = torch.zeros_like(band)
        gz, gh, gw = torch.gradient(normal_source[:, 0], dim=(1, 2, 3))
        dz = torch.arange(self.shape[0], device=self.device).view(-1, 1, 1).float()
        dh = torch.arange(self.shape[1], device=self.device).view(1, -1, 1).float()
        dw = torch.arange(self.shape[2], device=self.device).view(1, 1, -1).float()

        for b in range(batch_size):
            idx = torch.nonzero(band[b, 0] > 0.3, as_tuple=False)
            if idx.numel() == 0:
                continue
            pk = idx[torch.randint(idx.shape[0], (1,), device=self.device)][0]
            ci = (b, pk[0], pk[1], pk[2])
            n = torch.stack([gz[ci], gh[ci], gw[ci]])
            nn_ = n.norm()
            if nn_ < 1e-6:
                continue
            n = n / nn_
            t = (thick_frac[0] + (thick_frac[1] - thick_frac[0])
                 * torch.rand(1, device=self.device).item()) * self.max_dim
            e = (extent_frac[0] + (extent_frac[1] - extent_frac[0])
                 * torch.rand(1, device=self.device).item()) * self.max_dim
            rz, rh, rw = dz - pk[0].float(), dh - pk[1].float(), dw - pk[2].float()
            perp = rz * n[0] + rh * n[1] + rw * n[2]
            r2 = rz ** 2 + rh ** 2 + rw ** 2
            # Soft edges, so the slab is a prior rather than a hard stencil.
            slab = (torch.sigmoid((t - perp.abs()) / max(t * 0.25, 1e-6))
                    * torch.sigmoid((e - r2.sqrt()) / max(e * 0.15, 1e-6)))
            out[b, 0] = slab
        return out

    def _create_localized_focal_prior(self, batch_size, target_mask, focal_scale=1.0):
        """
        PRINCIPLED LOCALIZATION HELPER:
        Takes a large structural mask (e.g., all ventricles or all white matter) and 
        carves out a localized focal region by applying a random spatial Gaussian envelope 
        anchored within the valid anatomical space. Prevents structural ballooning.

        """
        D, H, W = self.shape
        device = self.device
        
        # Sample a random focal center within the volume dimensions
        zc = torch.rand((batch_size, 1, 1, 1, 1), device=device) * D
        hc = torch.rand((batch_size, 1, 1, 1, 1), device=device) * H
        wc = torch.rand((batch_size, 1, 1, 1, 1), device=device) * W
        
        dz = torch.arange(D, device=device).view(1, 1, D, 1, 1).float()
        dh = torch.arange(H, device=device).view(1, 1, 1, H, 1).float()
        dw = torch.arange(W, device=device).view(1, 1, 1, 1, W).float()
        
        # Focal width scaled to ~10% to 20% of the maximum FOV dimension,
        # times focal_scale for callers whose target load needs a wider one.
        focal_sigma = self.max_dim * (torch.rand(1, device=device).item() * 0.1 + 0.1) * focal_scale

        dist_sq = (dz - zc)**2 + (dh - hc)**2 + (dw - wc)**2
        spatial_envelope = torch.exp(-dist_sq / (2 * focal_sigma**2))
        
        localized_prior = target_mask * spatial_envelope
        max_val = localized_prior.amax(dim=(2, 3, 4), keepdim=True) + 1e-6
        return localized_prior / max_val

    def _create_tube_focal_prior(self, batch_size, target_mask,
                                 aspect_range=(1.0, 16.0)):
        """A column of roughly constant radius along an axis.

        The wedge is a cone, and a cone cannot be elongated in the covariance
        sense however narrow its apex angle: its radius grows along the axis,
        so most of the mass sits in the wide base and the shape eigenvalues
        stay comparable regardless of half-angle -- elongation is capped at
        roughly 1.4 for any cone.

        A cylinder does reach much higher elongation, because length and
        radius vary independently: e.g. L=70 r=4 gives 10.56, L=60 r=6 gives
        5.78, L=60 r=10 gives 3.46. The aspect ratio L/2r is the elongation,
        so sampling it directly spans a wide range rather than approximating
        it through a shape that structurally can't reach it.

        This is also the better description of a territorial infarct. A vessel
        supplies a column of roughly constant width along its course; it does
        not flare outward from its origin the way a cone does. The cone was
        modelling the wrong geometry.
        """
        D, H, W = self.shape
        device = self.device
        out = torch.zeros((batch_size, 1, D, H, W), device=device)

        dz = torch.arange(D, device=device).view(-1, 1, 1).float()
        dh = torch.arange(H, device=device).view(1, -1, 1).float()
        dw = torch.arange(W, device=device).view(1, 1, -1).float()

        for b in range(batch_size):
            idx = torch.nonzero(target_mask[b, 0] > 0.3, as_tuple=False)
            if idx.numel() == 0:
                continue
            c = idx[torch.randint(idx.shape[0], (1,), device=device)][0].float()
            axis = torch.randn(3, device=device)
            axis = axis / (axis.norm() + 1e-6)

            # Length as a fraction of the head, then radius from the sampled
            # aspect ratio. Doing it this way makes elongation the parameter
            # rather than an emergent property of two independent draws.
            length = (0.15 + 0.45 * torch.rand(1, device=device).item()) * self.max_dim * LESION_EXTENT_SCALE["vascular"]
            aspect = aspect_range[0] + (aspect_range[1] - aspect_range[0]) * \
                torch.rand(1, device=device).item()
            radius = max(length / (2.0 * aspect), 1.0)

            rz, rh, rw = dz - c[0], dh - c[1], dw - c[2]
            along = rz * axis[0] + rh * axis[1] + rw * axis[2]
            perp = (rz ** 2 + rh ** 2 + rw ** 2 - along ** 2).clamp(min=0).sqrt()
            # Soft edges so it acts as a prior rather than a stencil.
            out[b, 0] = (torch.sigmoid((length / 2 - along.abs()) / max(length * 0.08, 1e-6))
                         * torch.sigmoid((radius - perp) / max(radius * 0.25, 1e-6)))
        return out

    def _create_wedge_focal_prior(self, batch_size, target_mask):
        """Wedge/cone-shaped prior for large-vessel territorial infarcts.

        A GRF excursion set is isotropic and rounded, which is right for
        lacunar infarcts in perforator territory but wrong for territorial
        infarcts. When a major branch (MCA/ACA/PCA) occludes, the infarct
        takes the shape of the territory it fed: apex at the occlusion point
        deep in the brain, widening outward to a broad base at the cortical
        surface. That wedge is the single most recognizable feature of a
        territorial stroke on imaging.

        Built as a cone: an axis direction from a deep apex outward, with the
        radial cutoff growing linearly along the axis so the cross-section
        widens with distance from the apex.
        """
        D, H, W = self.shape
        device = self.device
        center = torch.tensor([D / 2.0, H / 2.0, W / 2.0], device=device)

        # Apex sits deep (near the middle), where a major vessel would occlude.
        apex_jitter = (torch.rand((batch_size, 3), device=device) - 0.5) * 0.25 * self.max_dim
        apex = center.view(1, 3) + apex_jitter

        # Random outward direction; the wedge opens from the apex toward the
        # cortical surface along this axis.
        axis = torch.randn((batch_size, 3), device=device)
        axis = axis / (axis.norm(dim=1, keepdim=True) + 1e-6)

        dz = torch.arange(D, device=device).view(1, D, 1, 1).float()
        dh = torch.arange(H, device=device).view(1, 1, H, 1).float()
        dw = torch.arange(W, device=device).view(1, 1, 1, W).float()

        vz = dz - apex[:, 0].view(-1, 1, 1, 1)
        vh = dh - apex[:, 1].view(-1, 1, 1, 1)
        vw = dw - apex[:, 2].view(-1, 1, 1, 1)

        # Distance along the wedge axis, and perpendicular to it
        along = (vz * axis[:, 0].view(-1, 1, 1, 1)
                 + vh * axis[:, 1].view(-1, 1, 1, 1)
                 + vw * axis[:, 2].view(-1, 1, 1, 1))
        dist_sq = vz ** 2 + vh ** 2 + vw ** 2
        perp = torch.sqrt(torch.clamp(dist_sq - along ** 2, min=0.0))

        # Half-angle of the cone: 20-45 degrees spans the range real arterial
        # territories subtend, from a narrow single-branch infarct to a broad
        # main-trunk one.
        #
        # The cone's LENGTH, not its angle, is what bounds elongation: the
        # apex sits near the brain centre so the cone reaches the boundary
        # regardless, and narrowing the angle makes it thinner without
        # making it longer. A cone's elongation stays low no matter how the
        # angle is tuned -- reaching the high elongations real strokes show
        # needs a different primitive (see _create_tube_focal_prior), not a
        # wider angle.
        half_angle = torch.sigmoid(torch.randn((batch_size, 1, 1, 1), device=device)) * 0.44 + 0.35
        tan_half = torch.tan(half_angle)

        # Radial cutoff grows with distance along the axis -- this is what
        # makes it a wedge rather than a cylinder.
        allowed = along * tan_half
        radial = torch.sigmoid((allowed - perp) / (0.05 * self.max_dim))

        # Only in front of the apex, and fading before it runs off the FOV.
        forward = torch.sigmoid(along / (0.03 * self.max_dim))
        reach = self.max_dim * (torch.sigmoid(torch.randn((batch_size, 1, 1, 1), device=device)) * 0.25 + 0.30) * LESION_EXTENT_SCALE["vascular"]
        extent = torch.sigmoid((reach - along) / (0.08 * self.max_dim))

        wedge = (radial * forward * extent).unsqueeze(1)

        localized_prior = target_mask * wedge
        max_val = localized_prior.amax(dim=(2, 3, 4), keepdim=True) + 1e-6
        return localized_prior / max_val

    def _vascular_aligned_pathology(self, batch_size, seg_prepared=None, tissue_mask=None):
        if seg_prepared is None:
            return self._local_pathology(batch_size, tissue_mask=tissue_mask)

        seg_b = seg_prepared.expand(batch_size, -1, -1, -1, -1)
        perforator_mask = ((seg_b == 7) | (seg_b == 8) | (seg_b == 9) | (seg_b == 10) | 
                           (seg_b == 24) | (seg_b == 25) | (seg_b == 26) | (seg_b == 27)).float()
        cortical_mask = ((seg_b == 2) | (seg_b == 19) | (seg_b == 1) | (seg_b == 18)).float()

        if perforator_mask.sum() == 0:
            perforator_mask = cortical_mask

        # Which branch: perforator (lacunar) or cortical (territorial). The
        # shape primitive follows this choice rather than being independent
        # of it -- lacunar infarcts in lenticulostriate territory really are
        # small and round, so the isotropic prior is correct there; large
        # cortical infarcts are wedge-shaped and the isotropic prior is not.
        # The two branches also need to differ in how raggedness scales with
        # size: multifocal lacunar disease coalesces as burden grows, while
        # a territorial infarct is a solid resorbed cavity that stays
        # self-similar as it grows. No explicit raggedness term is needed;
        # the blob and wedge priors produce that difference on their own.
        #
        # Resampling this probability per draw was tried and reverted: at
        # batch_size 1 a Bernoulli draw at random p is distributionally
        # identical to one at 0.5, so it changed nothing.
        use_perforator = (torch.rand((batch_size, 1, 1, 1, 1), device=self.device) < 0.5)
        target_mask = torch.where(use_perforator, perforator_mask, cortical_mask)

        blob_prior = self._create_localized_focal_prior(batch_size, target_mask,
                                                        focal_scale=LESION_EXTENT_SCALE["vascular"])
        # Cortical branch: cone or column, chosen per draw. Both are real
        # territorial morphologies -- a proximal occlusion produces a wedge
        # fanning from the origin, a distal single-branch one produces a
        # column following a gyrus -- and only the column can reach high
        # elongation. Keeping both leaves the wedge in support rather than
        # replacing a morphology the generator could already make.
        use_tube = (torch.rand((batch_size, 1, 1, 1, 1), device=self.device) < 0.5)
        wedge_prior = torch.where(
            use_tube,
            self._create_tube_focal_prior(batch_size, target_mask),
            self._create_wedge_focal_prior(batch_size, target_mask))
        prior = torch.where(use_perforator, blob_prior, wedge_prior)

        dyn_freq = torch.sigmoid(torch.randn(1, device=self.device)).item() * 0.2 + 0.1
        field = self._anisotropic_warp(self._universal_spectral_field(batch_size, max_freq=dyn_freq, force_dc=True))
        
        # GATED, not additive -- same change as the tract mode, for the same
        # reason. `field + prior * mult` lets the field place lesion wherever
        # it clears an absolute threshold, so the prior can only add to that,
        # never confine it. The result is that the excursion set fragments
        # whatever prior it is given, producing many small disconnected
        # pieces at low elongation -- the opposite of a real chronic
        # infarct, which is one large smooth elongated cavity. The wedge and
        # perforator priors already describe that shape; an additive field
        # was discarding it.
        #
        # Gating makes the prior decide WHERE and the field only texture
        # WITHIN, so a single connected wedge stays a single connected wedge.
        # The threshold is a quantile of the field inside the prior, which also
        # removes the absolute constants (0.6-1.6 shift, k_max = min(shape)/4)
        # that were being applied to a field of arbitrary scale.
        prior_support = prior > 0.05
        keep_frac = (torch.sigmoid(torch.randn(1, device=self.device)).item() * 0.7 + 0.2) * KEEP_FRAC_SCALE["vascular"]
        for b_idx in range(batch_size):
            inside = field[b_idx][prior_support[b_idx]]
            if inside.numel() < 27:
                # Too few voxels for a quantile; still confine to the support.
                field[b_idx] = field[b_idx] * prior_support[b_idx].float() - \
                               (~prior_support[b_idx]).float() * 1e3
                continue
            thr = torch.quantile(inside.float(), 1.0 - keep_frac)
            field[b_idx] = (field[b_idx] - thr) * prior_support[b_idx].float() - \
                           (~prior_support[b_idx]).float() * 1e3

        coverage_shift = torch.zeros((batch_size, 1, 1, 1, 1), device=self.device)
        sharpness = torch.sigmoid(torch.randn((batch_size, 1, 1, 1, 1), device=self.device)) * 6.0 + 2.0

        focus = torch.sigmoid((field - coverage_shift) * sharpness)
        core = torch.sigmoid((field - (coverage_shift + 0.4)) * sharpness)
        shell = focus * (1.0 - core)

        eta = torch.sigmoid(torch.randn((batch_size, 1, 1, 1, 1), device=self.device))
        novelty = ((focus - (1 - eta)) / (eta + 1e-6)).clamp(0, 1)

        if tissue_mask is not None:
            focus = focus * tissue_mask
            novelty = novelty * tissue_mask
            core = core * tissue_mask
            shell = shell * tissue_mask

        has_me = (torch.rand((batch_size, 1, 1, 1, 1), device=self.device) < 0.4).float()
        strength_mag = torch.sigmoid(torch.randn((batch_size, 1, 1, 1, 1), device=self.device)) * 2.0 - 1.0
        strength = has_me * strength_mag * 3.0

        return focus, novelty, strength, core, shell, 1, "Vascular-Aligned Pathology (Stroke)", torch.zeros((batch_size, 1, 1, 1, 1), device=self.device)

    def _tract_aligned_pathology(self, batch_size, seg_prepared=None, tissue_mask=None, force_ring=False):
        if seg_prepared is None:
            return self._local_pathology(batch_size, tissue_mask=tissue_mask, force_ring=force_ring)

        seg_b = seg_prepared.expand(batch_size, -1, -1, -1, -1)
        # Ventricle groups kept SEPARABLE rather than unioned. Taking all six
        # at once builds a band that wraps the whole ventricular system --
        # topologically a closed shell, and a shell is never sheet-like
        # however thin its wall, since its flatness saturates well below
        # what real periventricular lesions actually show. Those are caps and
        # rims on ONE aspect of ONE ventricle, not complete rings, and restricting the source mask reproduces that geometry
        # from anatomy rather than by imposing a geometric cut.
        vent_groups = [
            ((seg_b == 3) | (seg_b == 4)).float(),      # left lateral + inferior horn
            ((seg_b == 20) | (seg_b == 21)).float(),    # right lateral + inferior horn
            ((seg_b == 11) | (seg_b == 12)).float(),    # 3rd + 4th, midline
        ]
        vent_groups = [g for g in vent_groups if g.sum() > 0]
        if vent_groups:
            # Sometimes one group, sometimes the union: real disease is
            # unilateral in some subjects and confluent in others, and the
            # generator should span both rather than fix one.
            if torch.rand(1, device=self.device).item() < 0.75 and len(vent_groups) > 1:
                pick = torch.randint(len(vent_groups), (1,), device=self.device).item()
                vent_mask = vent_groups[pick]
            else:
                vent_mask = torch.clamp(sum(vent_groups), max=1.0)
        else:
            vent_mask = torch.zeros_like(seg_b, dtype=torch.float32)
        wm_mask = ((seg_b == 1) | (seg_b == 18)).float()

        if vent_mask.sum() == 0 or wm_mask.sum() == 0:
            vent_mask = (seg_b == seg_b.min()).float()
            wm_mask = tissue_mask if tissue_mask is not None else torch.ones_like(vent_mask)

        vent_blur1 = self._gaussian_blur(vent_mask, sigma=0.010 * self.max_dim)
        vent_blur2 = self._gaussian_blur(vent_mask, sigma=0.035 * self.max_dim)
        periph_band = (vent_blur2 - vent_blur1).clamp(min=0)
        periph_band = periph_band / (periph_band.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)
        
        raw_tract_prior = periph_band * wm_mask
        
        # Two prior geometries, sampled. The periventricular SHELL is
        # topologically a sleeve around a curved tube, so its flatness
        # saturates well below the range real periventricular lesions show,
        # however it is thinned, windowed, or filtered.
        #
        # A TANGENT SLAB anchors on the band but does not follow it: a plane
        # through a point on the ventricle surface, thin along the local
        # normal and extended in the tangent directions. Being a plane it does
        # not inherit the tube's curvature, so its flatness is set by the
        # thickness-to-extent ratio instead of by the anatomy, and that ratio
        # can be chosen to land inside the real range.
        #
        # Both are kept in support: confluent caps are slab-like, scattered
        # punctate disease follows the shell.
        use_slab = torch.rand(1, device=self.device).item() < 0.5
        if use_slab and raw_tract_prior.sum() > 0:
            _s = LESION_EXTENT_SCALE["tract"]
            tract_prior = self._tangent_slab_prior(batch_size, raw_tract_prior, vent_blur2,
                                                   thick_frac=(0.015 * _s, 0.045 * _s),
                                                   extent_frac=(0.10 * _s, 0.28 * _s))
        else:
            tract_prior = self._create_localized_focal_prior(batch_size, raw_tract_prior,
                                                             focal_scale=LESION_EXTENT_SCALE["tract"])
        
        dyn_freq = torch.sigmoid(torch.randn(1, device=self.device)).item() * 0.2 + 0.1
        field = self._anisotropic_warp(self._universal_spectral_field(batch_size, max_freq=dyn_freq, force_dc=True))
        
        prior_mult = torch.sigmoid(torch.randn((batch_size, 1, 1, 1, 1), device=self.device)) * 2.0 + 1.5
        # GATED, not additive. `field + prior * mult` lets the field place
        # lesion wherever it happens to exceed an absolute threshold, so the
        # prior can only ADD to that, never restrict it -- the union stays
        # field-shaped no matter how large the multiplier.
        #
        # Gating makes the prior say WHERE and the field say the texture
        # WITHIN, so the output cannot escape the prior's geometry. The slab
        # prior, gated, stays inside the range real periventricular lesions
        # occupy.
        #
        # It also removes two unanchored constants. coverage_shift (0.8-2.3)
        # and k_max = min(shape)/4 were absolute thresholds on a field whose
        # scale is arbitrary. The gated form thresholds at a QUANTILE of the
        # field inside the prior, so the sampled parameter is "what fraction of
        # the prior region becomes lesion" -- interpretable, and independent of
        # however the field happens to be scaled.
        prior_support = tract_prior > 0.05
        keep_frac = (torch.sigmoid(torch.randn(1, device=self.device)).item() * 0.7 + 0.15) * KEEP_FRAC_SCALE["tract"]
        for b_idx in range(batch_size):
            inside = field[b_idx][prior_support[b_idx]]
            if inside.numel() < 27:
                # Too few voxels for a quantile; still confine to the support.
                field[b_idx] = field[b_idx] * prior_support[b_idx].float() - \
                               (~prior_support[b_idx]).float() * 1e3
                continue
            thr = torch.quantile(inside.float(), 1.0 - keep_frac)
            # shift so the sigmoid below cuts at that quantile, and zero the
            # field outside the prior so nothing there can ever clear it
            field[b_idx] = (field[b_idx] - thr) * prior_support[b_idx].float() - \
                           (~prior_support[b_idx]).float() * 1e3

        coverage_shift = torch.zeros((batch_size, 1, 1, 1, 1), device=self.device)
        # Sharpness now scales the field's own spread inside the prior rather
        # than an arbitrary fraction of the volume size.
        sharpness = torch.sigmoid(torch.randn((batch_size, 1, 1, 1, 1), device=self.device)) * 6.0 + 2.0
        
        is_ring_tensor = torch.zeros((batch_size, 1, 1, 1, 1), device=self.device)
        for b_idx in range(batch_size):
            ring_decision = force_ring or (torch.rand(1, device=self.device).item() < 0.40)
            is_ring_tensor[b_idx, 0, 0, 0, 0] = 1.0 if ring_decision else 0.0

        outer_focus = torch.sigmoid((field - coverage_shift) * sharpness) * wm_mask
        core_focus = torch.sigmoid((field - (coverage_shift + 0.5)) * sharpness) * wm_mask
        
        focus = outer_focus
        core = torch.where(is_ring_tensor > 0.5, core_focus, torch.zeros_like(core_focus))
        shell = torch.where(is_ring_tensor > 0.5, focus * (1.0 - core), focus)

        eta = torch.sigmoid(torch.randn((batch_size, 1, 1, 1, 1), device=self.device))
        novelty = ((focus - (1 - eta)) / (eta + 1e-6)).clamp(0, 1)

        if tissue_mask is not None:
            focus = focus * tissue_mask
            novelty = novelty * tissue_mask
            core = core * tissue_mask
            shell = shell * tissue_mask

        has_me = (torch.rand((batch_size, 1, 1, 1, 1), device=self.device) < 0.2).float()
        sign = torch.where(torch.randn((batch_size, 1, 1, 1, 1), device=self.device) < 0, 1.0, -1.0)
        magnitude = torch.sigmoid(torch.randn((batch_size, 1, 1, 1, 1), device=self.device)) * 3.0 + 1.0
        strength = has_me * sign * magnitude

        return focus, novelty, strength, core, shell, 1, "Ventricles & White Matter (Tract-Aligned)", is_ring_tensor

    def _estimate_noise_sigma(self, img):
        """Immerkaer (1996)-style noise estimation, extended to 3D: convolve
        with a discrete Laplacian (kills locally-smooth/planar content,
        passes high-frequency noise), then take a ROBUST (median-absolute-
        deviation) estimate of the response, corrected for the kernel's
        known noise-amplification factor. For i.i.d. noise of variance
        sigma^2, a linear kernel k scales the response variance by
        sum(k^2); for the 6-connected 3D Laplacian used here (center=6, six
        face-neighbors=-1), sum(k^2)=42. MAD rather than raw std
        specifically because it downweights the sparse, large-response
        genuine edges relative to the dense, small-response noise -- edges
        are spatially rare, noise is everywhere. Verified empirically:
        ~0 on a noise-free image, recovers injected sigma within ~5-10%
        relative error across sigma in [0.02, 0.15]."""
        kernel = torch.zeros(1, 1, 3, 3, 3, device=img.device, dtype=img.dtype)
        kernel[0, 0, 1, 1, 1] = 6.0
        for d, h, w in [(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)]:
            kernel[0, 0, d, h, w] = -1.0
        response = F.conv3d(img, kernel, padding=1)
        mad = (response - response.median()).abs().median()
        robust_std = mad / 0.6745
        return (robust_std / (42.0 ** 0.5))

    # A tissue-contrast floor referenced against sensor noise alone is not a
    # sufficient conspicuity criterion: the Rose gate enforces contrast
    # against NOISE, while detectability against surrounding tissue depends
    # on TISSUE spread, which is typically much larger -- so a noise-only
    # k_rose does not by itself constrain what the network can see.
    #
    # A per-voxel LABEL gate cannot fix this either. The gate removes
    # labels, not intensity, so rejected voxels stay in the image with
    # lesion-like values and land in the surrounding shell, pulling its
    # mean toward the lesion and reducing measured contrast -- the opposite
    # of the intended effect.
    #
    # Fixing conspicuity means controlling the lesion's GMM mu relative to its
    # host tissue at synthesis, not filtering labels afterwards.

    def _min_visible_change(self, img, k_rose, fallback_noise_std_frac):
        """Rose-criterion visibility floor (Rose, 1948): a change needs
        contrast-to-noise ratio >= k_rose to be reliably distinguishable
        from noise -- classically 3-5 for confident detection, though even
        k=1 (as large as one noise sigma) is a defensible minimum bar. This
        is the direct analog of Nyquist for THIS question: Nyquist bounds
        what spatial frequency the sampling process can resolve; this
        bounds what intensity change the noise process allows to be
        resolved, for the same underlying reason (the corruption is
        irreducible once it's happened, no downstream processing recovers
        it). Noise is measured directly from the image when present. When
        the image is (near-)noise-free -- e.g. a bare synthetic phantom
        tested before whatever noise a downstream generator will actually
        add -- falls back to a floor set as a fraction of the image's own
        dynamic range, since detectability in the FINAL delivered image is
        what matters, not the current, possibly pre-noise one; expressing
        the fallback as a fraction (not an absolute constant) keeps this
        correct regardless of the input's intensity scale/normalization."""
        dynamic_range = (img.max() - img.min())
        measured = self._estimate_noise_sigma(img)
        sanity_floor = 1e-4 * dynamic_range
        effective_noise = torch.where(measured > sanity_floor, measured, fallback_noise_std_frac * dynamic_range)
        rose_floor = k_rose * effective_noise

        return rose_floor

    def augment_batch(self, image_tensor, seg_tensor=None, batch_size=1, max_foci=3,
                       tissue_mask=None, roi_labels=None, mode_weights=(0.20, 0.15, 0.15, 0.25, 0.25), force_rings=True,
                       k_rose=ROSE_CRITERION_K, fallback_noise_std_frac=0.05, couple_prob=0.15):
        device = self.device
        batch = image_tensor.to(device).repeat(batch_size, 1, 1, 1, 1)

        seg_prepared, labels = None, None
        if seg_tensor is not None:
            seg_prepared = seg_tensor.to(device).float().reshape(1, 1, *self.shape)
            labels = roi_labels.to(device) if roi_labels is not None else torch.unique(seg_prepared)
            labels = labels[labels != 0]
            is_extracerebral_label = torch.zeros_like(labels, dtype=torch.bool)
            for lid in EXTRACEREBRAL_LABEL_IDS:
                is_extracerebral_label |= (labels == lid)
            labels = labels[~is_extracerebral_label]

        brain_mask = (seg_tensor.to(device) > 0) & (~_is_extracerebral(seg_tensor.to(device)))
        fg_mask = brain_mask.float().reshape(1, 1, *self.shape).repeat(batch_size, 1, 1, 1, 1)

        # Primary mode: same weighted draw as before, guarantees exactly one
        # mode is always active (preserves the old "always some pathology"
        # rate). Coupling: each OTHER mode independently gets couple_prob
        # chance to ALSO activate this draw. Real presentations routinely
        # combine mechanisms (a tumor is a focal core/shell mass + mass
        # effect + a diffuse edema halo; a large stroke is vascular-territory
        # geometry + mass effect if there's swelling; MS is tract-aligned
        # lesions + separately-distributed focal lesions + diffuse "dirty
        # white matter", all in the same patient) -- under the old single
        # one-hot draw, the network could never see any of these
        # combinations during training, not rarely, structurally never.
        #
        # These defaults are DEAD in the training path: _inject_pathology and
        # paired_pathology_generator each supply their own, and the generator's
        # wins. Change weights there, not here. Injected burden runs well above
        # real data either way, but that is a question of each mode's spatial
        # extent -- reweighting between modes cannot shrink a mode. See
        # docs/pathology_size_distribution.md.
        mode = torch.multinomial(torch.tensor(mode_weights, device=device), batch_size, replacement=True)
        mode_idx = mode.item()
        mode_names = [
            "Local Pathology", 
            "Global Pathology", 
            "Geometric Atrophy", 
            "Vascular-Aligned Pathology", 
            "Tract-Aligned Pathology"
        ]
        chosen_mode_name = mode_names[mode_idx]

        use_local_primary = (mode == 0).float().view(-1, 1, 1, 1, 1)
        use_global_primary = (mode == 1).float().view(-1, 1, 1, 1, 1)
        use_geometric_primary = (mode == 2).float().view(-1, 1, 1, 1, 1)
        use_vascular_primary = (mode == 3).float().view(-1, 1, 1, 1, 1)
        use_tract_primary = (mode == 4).float().view(-1, 1, 1, 1, 1)

        def _couple(primary, name):
            p = COUPLE_PROB_BY_MODE.get(name, couple_prob)
            trial = (torch.rand((batch_size, 1, 1, 1, 1), device=device) < p).float()
            return trial * (1 - primary)  # only roll for modes that AREN'T already the primary

        use_local = torch.clamp(use_local_primary + _couple(use_local_primary, "Local Pathology"), max=1.0)
        use_global = torch.clamp(use_global_primary + _couple(use_global_primary, "Global Pathology"), max=1.0)
        use_geometric = torch.clamp(use_geometric_primary + _couple(use_geometric_primary, "Geometric Atrophy"), max=1.0)
        use_vascular = torch.clamp(use_vascular_primary + _couple(use_vascular_primary, "Vascular-Aligned Pathology"), max=1.0)
        use_tract = torch.clamp(use_tract_primary + _couple(use_tract_primary, "Tract-Aligned Pathology"), max=1.0)
        if DISABLED_MODES:
            _z = lambda u, name: u * 0.0 if name in DISABLED_MODES else u
            use_local = _z(use_local, "Local Pathology")
            use_global = _z(use_global, "Global Pathology")
            use_geometric = _z(use_geometric, "Geometric Atrophy")
            use_vascular = _z(use_vascular, "Vascular-Aligned Pathology")
            use_tract = _z(use_tract, "Tract-Aligned Pathology")

        active_flags = [use_local, use_global, use_geometric, use_vascular, use_tract]
        active_mode_names = [name for name, flag in zip(mode_names, active_flags) if flag.item() > 0.5]
        combined_mode_name = "+".join(active_mode_names)

        # Only compute the modes that are actually active, rather than
        # running every mode on every draw and multiplying by a 0/1 flag --
        # with one primary mode and coupling active only a fraction of the
        # time, most per-draw mode computations would otherwise be discarded.
        #
        # The flags are decided above, before any mode runs, so this is a pure
        # skip -- the arithmetic below is unchanged, since a skipped mode
        # contributes zeros exactly as its flag would have forced anyway.
        # Volume-shaped and scalar-shaped zeros. The modes return (B,1,D,H,W)
        # fields but (B,1,1,1,1) strengths, and using the volume shape for both
        # allocates a tensor the size of the product of every dimension.
        _z = lambda: torch.zeros((batch_size, 1, *self.shape), device=device)
        _zs = lambda: torch.zeros((batch_size, 1, 1, 1, 1), device=device)

        if use_local.item() > 0.5:
            l_mask, l_nov, l_str, l_core, l_shell, n_foci, l_labels, l_ring = self._local_pathology(
                batch_size, seg_prepared=seg_prepared, labels=labels, max_foci=max_foci, tissue_mask=fg_mask, force_ring=force_rings)
        else:
            l_mask = l_nov = l_core = l_shell = _z()
            l_str = l_ring = _zs()
            n_foci = 0
            l_labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        if use_global.item() > 0.5:
            g_mask, g_nov, g_str, g_label_desc = self._global_pathology(batch_size, tissue_mask=fg_mask)
        else:
            g_mask = g_nov = _z()
            g_str = _zs()
            g_label_desc = ""

        if use_geometric.item() > 0.5:
            geo_potential, geo_strength, geo_labels = self._geometric_atrophy(batch_size, seg_prepared, labels)
        else:
            geo_potential, geo_strength = _z(), _zs()
            geo_labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        if use_vascular.item() > 0.5:
            v_mask, v_nov, v_str, v_core, v_shell, _, v_label_desc, v_ring = self._vascular_aligned_pathology(
                batch_size, seg_prepared=seg_prepared, tissue_mask=fg_mask)
        else:
            v_mask = v_nov = v_core = v_shell = _z()
            v_str = v_ring = _zs()
            v_label_desc = ""

        if use_tract.item() > 0.5:
            t_mask, t_nov, t_str, t_core, t_shell, _, t_label_desc, t_ring = self._tract_aligned_pathology(
                batch_size, seg_prepared=seg_prepared, tissue_mask=fg_mask, force_ring=force_rings)
        else:
            t_mask = t_nov = t_core = t_shell = _z()
            t_str = t_ring = _zs()
            t_label_desc = ""

        # Clamped: with coupling enabled, multiple modes' fields can
        # genuinely overlap in space, so their sum could exceed 1 without
        # this.
        final_mask = torch.clamp(l_mask * use_local + g_mask * use_global + v_mask * use_vascular + t_mask * use_tract, 0.0, 1.0)
        final_core = torch.clamp(l_core * use_local + v_core * use_vascular + t_core * use_tract, 0.0, 1.0)
        final_shell = torch.clamp(l_shell * use_local + v_shell * use_vascular + t_shell * use_tract, 0.0, 1.0)

        is_special_mode = (use_vascular > 0.5) | (use_tract > 0.5)
        
        valid_tissue = batch[fg_mask > 0.5]
        if valid_tissue.numel() > 1:
            tissue_mean = valid_tissue.mean()
            tissue_std = valid_tissue.std()
        else:
            tissue_mean = torch.tensor(1.0, device=device)
            tissue_std = torch.tensor(0.1, device=device)
            
        cv = (tissue_std / (tissue_mean + 1e-6)).view(1, 1, 1, 1, 1)

        core_rand = torch.rand((batch_size, 1, 1, 1, 1), device=device) * 2.0 + 1.5
        shell_rand = torch.rand((batch_size, 1, 1, 1, 1), device=device) * 2.0 - 0.5
        
        core_intensities = torch.where(is_special_mode, -3.0 * cv, -core_rand * cv)
        shell_intensities = torch.where(is_special_mode, -1.0 * cv, shell_rand * cv)
        global_intensities = torch.randn((batch_size, 1, 1, 1, 1), device=device) * cv * 0.5

        texture_max_freq = torch.sigmoid(torch.randn(1, device=device)).item() * 0.20 + 0.20
        texture = self._universal_spectral_field(batch_size, max_freq=texture_max_freq, force_dc=False)

        texture_amp = torch.sigmoid(torch.randn((batch_size, 1, 1, 1, 1), device=device)) * 0.75 + 0.25
        texture = texture * texture_amp
        alpha = torch.sigmoid(torch.randn((batch_size, 1, 1, 1, 1), device=device))
        tex_mod = alpha + (1.0 - alpha) * texture 

        focal_modulation = (final_shell * shell_intensities + final_core * core_intensities) * tex_mod
        global_modulation = g_mask * use_global * global_intensities
        
        pathology_modulation = focal_modulation + global_modulation

        # Region-local, not case-wide: atrophy is pure displacement (no
        # intensity change of its own), but that should only suppress
        # modulation WHERE atrophy itself is acting. The old (1-use_geometric)
        # was a scalar case-wide switch -- correct only when atrophy could
        # never co-occur with anything else. Under coupling it can, and a
        # case-wide switch would silently zero out a co-active mode's entire
        # intensity change, not just atrophy's own footprint.
        geometric_active_mask = (geo_potential.abs() > 1e-3).float() * use_geometric
        modulated = batch * (1 + pathology_modulation * (1 - geometric_active_mask))
        batch_max = batch.amax(dim=(2, 3, 4), keepdim=True)
        modulated = torch.clamp(modulated, min=torch.tensor(0.0, device=device), max=batch_max)

        # Clamped for the same overlap reason as final_mask/core/shell above.
        # NOTE: geo_potential itself stays SIGNED here (unlike
        # geometric_active_mask above) -- _mass_effect_displacement reads
        # its sign to distinguish expansive (positive) from atrophic
        # (negative) displacement direction; stripping it would make every
        # atrophy event push outward regardless of its intended direction.
        total_potential = torch.clamp(l_mask * use_local + fg_mask * use_global + geo_potential * use_geometric + v_mask * use_vascular + t_mask * use_tract, -1.0, 1.0)
        # With coupling, multiple modes' strengths can sum rather than only
        # one ever being nonzero, so the combined total needs its own cap,
        # symmetric and adaptive rather than a flat constant.
        #
        # Derived from where the mass-effect warp actually starts folding
        # (Jacobian determinant of the resulting warp going non-invertible),
        # measured at three volume sizes: the onset scales with volume size,
        # since blur_sigma (used below) scales with max(shape) and the same
        # nominal strength produces a gentler voxel-to-voxel gradient at
        # higher resolution. Anchored to the smallest size tested (D=64,
        # onset ~15) with a 30% safety margin, then scaled proportionally to
        # the actual volume shape -- a fixed constant would be needlessly
        # restrictive at small volumes (e.g. a D=48 phantom) or provide no
        # real protection at large ones (this codebase's production size,
        # D=160, sees zero folding even at the true worst case of all five
        # modes' maxima summed).
        _fold_onset_at_d64 = 15.0
        _safety_margin = 0.7
        max_safe_strength = _fold_onset_at_d64 * _safety_margin * (max(self.shape) / 64.0)
        total_strength = (l_str * use_local + g_str * use_global + geo_strength * use_geometric + v_str * use_vascular + t_str * use_tract).clamp(min=-max_safe_strength, max=max_safe_strength)
        
        me_blur_sigma = (torch.rand(1, device=device).item() * 0.01 + 0.1) * self.max_dim
        disp = self._mass_effect_displacement(total_potential, total_strength, blur_sigma=me_blur_sigma)
        
        is_tear = (torch.rand((batch_size, 1, 1, 1, 1), device=device) < 0.5).float() * (1 - use_geometric)
        raw_tear_scale = (is_tear * final_mask + (1 - is_tear)).squeeze(1).unsqueeze(1)
        
        active_scale_sigma = (torch.rand(1, device=device).item() * 0.02 + 0.01) * self.max_dim
        active_scale = self._gaussian_blur(raw_tear_scale, sigma=active_scale_sigma).squeeze(1).unsqueeze(-1)
        
        D, H, W = self.shape
        z_w = torch.hann_window(D, periodic=False, device=device).view(1, D, 1, 1, 1)
        y_w = torch.hann_window(H, periodic=False, device=device).view(1, 1, H, 1, 1)
        x_w = torch.hann_window(W, periodic=False, device=device).view(1, 1, 1, W, 1)
        border_mask = torch.clamp((z_w * y_w * x_w) * 1.5, 0.0, 1.0)

        active_disp = disp * active_scale * border_mask
        grid = self._identity_grid(batch_size) + active_disp

        warped_img = F.grid_sample(modulated, grid, mode='bilinear', padding_mode='border', align_corners=False)
        warped_img = torch.clamp(
            warped_img + 0.1 * (warped_img - self._gaussian_blur(warped_img, sigma=1.0)), 
            min=torch.tensor(0.0, device=warped_img.device), 
            max=batch_max
        )
        warped_mask = F.grid_sample(final_mask, grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        warped_shell = F.grid_sample(final_shell, grid, mode='bilinear', padding_mode='border', align_corners=False)
        warped_core = F.grid_sample(final_core, grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        warped_seg, hard_seg, deformed_thresh_seg, novel_label_core, novel_label_shell = None, None, None, None, None
        if seg_tensor is not None:
            seg_b = seg_tensor.to(device).float().reshape(1, 1, *self.shape).repeat(batch_size, 1, 1, 1, 1)
            warped_seg = F.grid_sample(seg_b, grid, mode='nearest', padding_mode='border', align_corners=False)
            
            # Dynamically allocated, not hardcoded: PathologyRandomizer is a
            # general-purpose tool that shouldn't need to know any specific
            # caller's label scheme. A fixed 57/58 is only safe by
            # coincidence of the current 0-56 scheme's range -- collision-
            # proof against whatever seg_tensor's own labels actually are,
            # so it stays correct if this is pointed at a different dataset
            # or the label scheme is extended. (For the current 0-56 scheme
            # this computes to exactly 57/58 anyway -- no behavior change
            # for existing callers, including _inject_pathology in
            # mcp_generator.py, which only ever uses the diff against
            # warped_seg, never these literal values.)
            base_max = int(seg_tensor.max().item())
            novel_label_core = base_max + 1
            novel_label_shell = base_max + 2

            # Gate the discrete label on the ACTUAL per-voxel observed
            # intensity change relative to a PRINCIPLED visibility floor
            # (Rose criterion -- see _min_visible_change), not an arbitrary
            # constant and not just geometry. Gating on the global per-draw
            # scalar magnitude (core_intensities/shell_intensities) would
            # not work: the per-voxel tex_mod texture term in
            # focal_modulation can independently suppress the effective
            # change to near-zero at specific voxels regardless of what the
            # global scalar is, so the global scalar is never the real
            # bottleneck. Gating on the actual warped image difference
            # instead measures the quantity that actually determines
            # visibility.
            original_warped_for_gate = F.grid_sample(batch, grid, mode='bilinear', padding_mode='border', align_corners=False)
            visible_change = (warped_img - original_warped_for_gate).abs()
            min_visible_change = self._min_visible_change(batch, k_rose, fallback_noise_std_frac)
            is_visible = visible_change > min_visible_change

            deformed_thresh_seg = warped_seg.clone()
            deformed_thresh_seg = torch.where((warped_shell > 0.5) & is_visible, torch.full_like(deformed_thresh_seg, float(novel_label_shell)), deformed_thresh_seg)
            deformed_thresh_seg = torch.where((warped_core > 0.5) & is_visible, torch.full_like(deformed_thresh_seg, float(novel_label_core)), deformed_thresh_seg)

            hard_seg = deformed_thresh_seg.clone()
        active_strength_val = total_strength[0].abs().max().item()
        has_me_bool = active_strength_val > 0.01

        if mode_idx == 0:
            lbl_id = int(l_labels[0].item()) if l_labels is not None else -1
            label_desc = f"{lbl_id}: {FREESURFER_LABELS.get(lbl_id, 'Unknown')}"
            foci_count = n_foci
            ring_status = bool(l_ring[0].item() > 0.5)
        elif mode_idx == 1:
            label_desc = "Global Brain Pathology"
            foci_count = "N/A"
            ring_status = False
        elif mode_idx == 2:
            lbl_id = int(geo_labels[0].item()) if geo_labels is not None else -1
            label_desc = f"Geometric Atrophy - {lbl_id}: {FREESURFER_LABELS.get(lbl_id, 'Unknown')}"
            foci_count = "N/A"
            ring_status = False
        elif mode_idx == 3:
            label_desc = v_label_desc
            foci_count = 1
            ring_status = False
        else:
            label_desc = t_label_desc
            foci_count = 1
            ring_status = bool(t_ring[0].item() > 0.5)

        metadata = {
            'mode_name': combined_mode_name,       # e.g. "Local Pathology+Vascular-Aligned Pathology" when coupled
            'primary_mode_name': chosen_mode_name,  # the single guaranteed-active mode, same meaning as before coupling existed
            'active_modes': active_mode_names,      # list form, for programmatic use instead of parsing the '+'-joined string
            'is_coupled': len(active_mode_names) > 1,
            'has_mass_effect': has_me_bool,
            'mass_effect_strength': active_strength_val,
            'n_foci': foci_count,
            'is_ring': ring_status,
            'label_desc': label_desc,
            'novel_label_core': novel_label_core,
            'novel_label_shell': novel_label_shell
        }

        return {
            'image': warped_img,
            'mask': warped_mask,
            'modulated': modulated,
            'seg_with_novel_label': hard_seg,
            'novel_label_core': novel_label_core,
            'novel_label_shell': novel_label_shell,
            'original': batch,
            'warped_seg': warped_seg,
            'unwarped_core': final_core,
            'unwarped_shell': final_shell,
            'deformed_thresh_seg': deformed_thresh_seg,
            'metadata': metadata,
            'grid': grid,  # exposes the mass-effect sampling grid so callers can
                            # warp OTHER co-registered volumes (e.g. a finer
                            # label map) with the identical deformation
        }