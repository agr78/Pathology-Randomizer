"""Super-resolution on REAL scans with REAL lesion masks, no painting (2026-10-10): the thesis test for the SR arms.

Held-out 1 mm scans (ATLAS, BraTS T1w: lesion masks; ABIDE: healthy, curated labels only) are degraded with the generator's own
slice model (util.make_mode_sampler's anisotropic case, fixed here to the WMH-like setting: one axis at --thick mm, blur std =
ln5/pi x thickness in mm, linear downsample to that spacing, linear zoom back to 1 mm, max-normalised) and super-resolved by each
arm exactly as sr_step does (input normalised to its own [p1, p99], residual added). Reported per scan and as medians:
    psnr_in / psnr_out      PSNR of the degraded input and of each arm's output against the original, inside the cerebral mask
    psnr_lesion / psnr_rest same, inside the real lesion mask and in the cerebral mask minus a 5-voxel lesion halo
    contrast_in / out / orig  mean lesion intensity minus mean of its 5-voxel halo: does the arm keep the lesion's contrast?
Degradation along the axis given by --axis (0 = the first array axis). Arms are loaded with td.load_task_model (feature-input or plain).
    cd <snapshot> && python eval/sr_eval.py --dataset ATLAS --ckpts A.pth B.pth --n 60 --thick 3 --out X.csv
2026-10-14: --bn_mode batch|running (default batch), the BatchNorm statistics of the loaded arms: batch = each volume normalised by
its own statistics (compare_datasets.bn_batch_mode), the convention of the detection readout and of the fine-tune validation;
running = the running averages (model.eval()), which reproduces the rows written before this flag. Trailing CSV column bn_mode.
--gt_dir: fallback GT derivative (<sub>_gt.nii.gz) when the dataset's own GT is not mounted, as register_eval.py.
"""
import argparse, csv, glob, os, sys, time
import numpy as np, nibabel as nib, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import train_downstream as td   # noqa: E402
from compare_datasets import list_cases, to_ras, EXTRACEREBRAL_COMPACT_IDX  # noqa: E402
from register_eval_pool import POOL, LUT  # noqa: E402

ROOTS = {ds: (os.environ.get('PR_DERIV_ROOT', '/path/to/deriv_root') + f"/{ds}/derivatives", os.environ.get('PR_EVAL_DATA_ROOT', '/path/to/eval_data_root') + f"/{ds}") for ds in ("ATLAS", "BraTS2023-GLI")}
ROOTS["WMH3DT1"] = ROOTS_W = (os.environ.get('PR_DERIV_ROOT', '/path/to/deriv_root') + "/WMH_seg_challenge/derivatives", os.environ.get('PR_EVAL_DATA_ROOT', '/path/to/eval_data_root') + "/WMH_seg_challenge")   # 2026-10-20: the native 1 mm 3D T1 of every WMH subject (60), GT rigidly resampled onto it (gt_3DT1) and a synthstrip brain mask (synthstrip_b0)
EX = torch.tensor(EXTRACEREBRAL_COMPACT_IDX)


def degrade(hr, axis, thick, dev, return_scale=False, cubic=False):
    """generator.py's resolution model for one anisotropic axis: Gaussian blur (std ln5/pi * thickness, in HR voxels), linear
    sampling at `thick` mm spacing, linear zoom back; max-normalised like `final`.
    return_scale (2026-10-14): also return that max. The generator divides the degraded image AND its high-resolution target by the
    SAME constant (generator.py:2673-2681: M_clean = max(resized_clean); final = resized/M; hr = HRbf/M, "on the degraded images'
    scale"), so the target must be scaled by the DEGRADED image's max, not by its own. Scoring hr/hr.max() put the target in a
    different frame from training: the blur lowers the peak, so normalising each volume by its own max makes the input systematically
    brighter than the target (brain mean 0.63 vs 0.51 on BraTS sub-00216) and the input PSNR 8.4 dB too low (17.4, not 25.8).
    cubic: ALSO return a cubic (order-3 B-spline) zoom-back of the
    SAME decimated stack, as a stronger interpolation baseline. Only the zoom-BACK differs -- the blur and the decimation are
    byte-identical and it is divided by the SAME M, so it lands in the frame the target is scaled into. F.interpolate has no 3-D
    cubic (its bicubic is 4-D only), hence scipy.ndimage.zoom(order=3). grid_mode=True is REQUIRED: that is the pixel-extent
    convention matching align_corners=False. Verified 2026-10-31 on a band-limited phantom: scipy order=1 grid_mode=True
    reproduces the trilinear zoom-back to 6e-08, and cubic then gains +2.35 dB over it, against +0.13 dB for the half-voxel
    shifted grid_mode=False -- i.e. the wrong grid makes the cubic baseline look spuriously weak instead of erroring.
    The NETWORK IS NOT FED THIS. Its input stays the trilinear zoom-back, which is the generator's own resolution model
    (generator.py:2704, same ln5/pi blur) and therefore the distribution every arm was trained on; a cubic input would be
    out of distribution on real data for nets trained on a trilinear one."""
    x = hr[None, None].float()
    std = float(np.log(5) / np.pi * thick)
    k = int(2 * round(3 * std) + 1); t = torch.arange(k, device=dev, dtype=x.dtype) - k // 2
    g = torch.exp(-0.5 * (t / std) ** 2); g = g / g.sum()
    shape = [1, 1, 1, 1, 1]; shape[2 + axis] = k
    pad = [0] * 6; pad[2 * (2 - axis)] = k // 2; pad[2 * (2 - axis) + 1] = k // 2
    xb = F.conv3d(F.pad(x, pad, mode="replicate"), g.view(shape))
    size = list(hr.shape); size[axis] = max(1, int(round(size[axis] / thick)))
    small = F.interpolate(xb, size=size, mode="trilinear", align_corners=False)
    back = F.interpolate(small, size=list(hr.shape), mode="trilinear", align_corners=False)[0, 0]
    M = back.max().clamp(min=1e-6)
    out = (back / M,)
    if return_scale: out = out + (M,)
    if cubic:
        from generator import bspline_resize_axis as _bra   # the SAME operator the generator uses under --bspline_zoom
        cu = small[0, 0]
        for _ax in range(3): cu = _bra(cu, _ax, int(list(hr.shape)[_ax]))
        assert list(cu.shape) == list(hr.shape), f"cubic zoom gave {tuple(cu.shape)}, expected {tuple(hr.shape)}"
        out = out + (cu / M,)
    return out if len(out) > 1 else out[0]


def psnr(a, b, m):
    mse = ((a - b) ** 2)[m].mean().clamp(min=1e-12); return float(10 * torch.log10(1.0 / mse))


def component_retention(gt_np, brain_np, x_in, x_out, min_vox=20, band=3, vis_sd=1.0):
    """LESION-RETENTION GATE: the output must not lose any lesion visible in the tricubic input. Per connected component of the real mask (>= min_vox voxels): contrast = mean inside - mean of a `band`-voxel halo (brain
    only, lesion excluded); a component is VISIBLE in the input when |contrast_in| >= vis_sd * sd(halo in the input). ratio = contrast_out /
    contrast_in (same sign). Returns dict(n, visible, lost95 = visible components whose ratio < 0.95, min_ratio, med_ratio) over VISIBLE ones."""
    from scipy import ndimage
    lab, n = ndimage.label(gt_np)
    if n == 0: return dict(n=0, visible=0, lost95=0, min_ratio=float("nan"), med_ratio=float("nan"))
    sizes = np.bincount(lab.ravel()); ratios = []; vis = 0
    for k in range(1, n + 1):
        if sizes[k] < min_vox: continue
        comp = lab == k; halo = ndimage.binary_dilation(comp, iterations=band) & ~gt_np & brain_np
        if halo.sum() < 10: continue
        c_in = float(x_in[comp].mean() - x_in[halo].mean()); sd = float(x_in[halo].std())
        if abs(c_in) < vis_sd * sd: continue
        vis += 1; c_out = float(x_out[comp].mean() - x_out[halo].mean()); ratios.append(c_out / c_in)
    if not ratios: return dict(n=int(n), visible=0, lost95=0, min_ratio=float("nan"), med_ratio=float("nan"))
    return dict(n=int(n), visible=vis, lost95=int(sum(r_ < 0.95 for r_ in ratios)), min_ratio=round(min(ratios), 4), med_ratio=round(float(np.median(ratios)), 4))


def visible_union(gt_np, brain_np, x_in, min_vox=20, band=3, vis_sd=1.0):
    """The union of the mask's connected components that are
    VISIBLE IN THE INPUT by component_retention's own rule (>= min_vox voxels, |mean inside - mean of the brain-only halo| >= vis_sd *
    sd(halo), measured on the NET INPUT x_in). psnr_*_lesion_vis columns score the lesion over this union only, so a FLAIR-drawn WMH
    mask on a T1 (~6 of ~76 components visible per case) is read where the contrast actually exists. Scoring-only: the input decides
    visibility, the target is never consulted. Returns (bool mask, number of visible components)."""
    from scipy import ndimage
    lab, n = ndimage.label(gt_np); vis = np.zeros_like(gt_np, dtype=bool); k_vis = 0
    if n == 0: return vis, 0
    sizes = np.bincount(lab.ravel())
    for k in range(1, n + 1):
        if sizes[k] < min_vox: continue
        comp = lab == k; halo = ndimage.binary_dilation(comp, iterations=band) & ~gt_np & brain_np
        if halo.sum() < 10: continue
        c_in = float(x_in[comp].mean() - x_in[halo].mean()); sd = float(x_in[halo].std())
        if abs(c_in) < vis_sd * sd: continue
        vis |= comp; k_vis += 1
    return vis, k_vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("ATLAS", "BraTS2023-GLI", "ABIDE", "WMH3DT1")); ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--thick", type=float, default=3.0); ap.add_argument("--axis", type=int, default=2)
    ap.add_argument("--labels", default="synthseg_healthy_98ea36c_synthseg_500000"); ap.add_argument("--gt_label", type=int, nargs="+", default=[1])
    ap.add_argument("--out", default=None); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--subjects", default=None, help="file of space-separated subject ids: restrict the cases (2026-10-13)")
    ap.add_argument("--chunk", type=int, nargs=2, default=None, metavar=("I", "K"), help="process every K-th case starting at I (array jobs)")
    ap.add_argument("--net_input", default="trilinear", choices=("trilinear", "cubic"), help="2026-10-31: which interpolation the NETWORK is fed. "
                    "trilinear = the zoom-back the arms were trained on (generator.py myzoom_torch, the default); cubic = the order-3 B-spline "
                    "zoom-back of the SAME decimated stack. 'cubic' is a TRAIN/TEST MISMATCH for arms trained on trilinear -- their residual "
                    "corrects trilinear error and cubic has already removed part of it -- so it is a diagnostic for whether an end-to-end cubic "
                    "pipeline is worth retraining, NOT a reportable configuration. Tagged |cubicin in the output columns so it cannot be "
                    "confused with the trilinear-input rows.")
    ap.add_argument("--lesion_gate", default="none", choices=("none", "gt", "detector", "unc"), help="2026-11-02 LESION-DEFICIT EXPERIMENTS. The nets "
                    "lose 0.9-1.5 dB to plain cubic interpolation INSIDE real lesions while beating it brain-wide, and in-domain the prior gains "
                    "+3.95 dB in synthetic lesions -- so the learned residual is wrong on real lesion appearance. This suppresses the residual "
                    "where a lesion is: gt = inside the ground-truth mask (an ORACLE upper bound of 'leave the lesion alone', never a result); "
                    "detector = weighted by the reported detector's own deviation map, w = clamp(|2*stored-1| / tau, 0, 1), pred = base + "
                    "(1-w)*res, which uses no ground truth and is a legitimate pipeline if it works. Columns are tagged |gate=... so they "
                    "cannot be confused with ungated rows.")
    ap.add_argument("--gate_deriv", default=None, help="derivatives subdir holding <sub>_reg.nii.gz for --lesion_gate detector, e.g. "
                    "synthseg_pathology_healthy_fork_joint_g500ema_s1_082da31_synthseg_500000_ema_bnbatch. A missing map FAILS the run.")
    ap.add_argument("--ext_dir", default=None, help="2026-11-05: directory of an EXTERNAL method's super-resolved volumes (dmprior sr: <id>_T1w_recon.nii.gz), scored in the arms' frame")
    ap.add_argument("--ext_pattern", default="{sub}_T1w_recon.nii.gz"); ap.add_argument("--ext_tag", default="dmprior_sr_T1w_only")
    ap.add_argument("--lesion_dilate", type=int, default=0, help="2026-11-05: dilate the lesion mask by N voxels (brain only) for the lesion PSNR columns; 0 = the mask itself")
    ap.add_argument("--gate_tau_unc", type=float, default=0.1, help="2026-11-10: --lesion_gate unc (uncertainty-head arms only): the residual is suppressed by w = clamp(b / tau, 0, 1), b the predicted Laplace scale in normalised intensity units; set tau from the CLEAN validation draws' b distribution (no real data), never from a scored set.")
    ap.add_argument("--gate_tau", type=float, default=0.4, help="deviation (network units, |2*stored-1|) at which the detector gate saturates to 1. "
                    "0.4 network units = 0.2 in stored units, the healthy-cohort-calibrated threshold.")
    ap.add_argument("--residual_norm", default="none", choices=("none", "offset", "affine"), help="2026-10-20: re-centre each arm's residual with the input's own brain statistics (offset: zero mean; affine: zero mean and no global gain along the input); tagged |<mode> in the columns")
    ap.add_argument("--gt_dir", default=None, help="fallback GT derivative (<sub>_gt.nii.gz) when the dataset's own GT path is not mounted (2026-10-14, as register_eval.py)")
    ap.add_argument("--bn_mode", default="batch", choices=("batch", "running"), help="BatchNorm statistics at inference (2026-10-14): batch = every BN layer normalises the volume being inferred by its own statistics (compare_datasets.bn_batch_mode, momentum 0; the convention of the detection readout and of the fine-tune validation; default), running = the checkpoint's running averages (model.eval(): the rows written before 2026-10-14). Recorded in every CSV row (trailing column bn_mode). Feature-input nets (FeatureHeadNet) are unaffected: trunk on batch statistics, head on running statistics in both modes.")
    a = ap.parse_args(); dev = torch.device(a.device); rng = np.random.default_rng(a.seed)
    if a.lesion_gate != "none" and a.dataset == "ABIDE": raise SystemExit("--lesion_gate needs a lesion dataset (ABIDE has no lesions and no derivatives root)")
    if a.lesion_gate == "detector" and not a.gate_deriv: raise SystemExit("--lesion_gate detector needs --gate_deriv")
    if a.dataset == "ABIDE":
        files = sorted(glob.glob(os.path.join(POOL, "ABIDE*T1w.nii"))); cases = [(os.path.basename(f).split(".")[0], f, None) for f in files]
    else:
        droot, data_dir = ROOTS[a.dataset]
        if a.dataset == "WMH3DT1":
            if a.labels == "synthseg_healthy_98ea36c_synthseg_500000": a.labels = "synthstrip_b0"   # the brain mask on the 3DT1 grid (binary: brain = label > 0)
            cases = [(f"{os.path.basename(os.path.dirname(os.path.dirname(t1)))}_{os.path.basename(os.path.dirname(t1))}_3DT1", t1,
                      os.path.join(droot, "gt_3DT1", f"{os.path.basename(os.path.dirname(os.path.dirname(t1)))}_{os.path.basename(os.path.dirname(t1))}_3DT1_gt.nii.gz"))
                     for t1 in sorted(glob.glob(os.path.join(data_dir, "*", "sub-*", "3DT1.nii.gz")))]
            cases = [c for c in cases if os.path.isfile(c[2])]
        else: cases = list_cases(data_dir, a.dataset)
        if not cases and a.gt_dir:   # container: the datasets' GT lives on an unmounted path; --gt_dir points at a copy under derivatives (2026-10-14)
            for t1 in sorted(glob.glob(os.path.join(data_dir, "sub-*", "anat", "*_T1w.nii.gz"))):
                sub = os.path.basename(t1).split("_")[0]; g = os.path.join(droot, a.gt_dir, f"{sub}_gt.nii.gz")
                if os.path.isfile(g): cases.append((sub, t1, g))
    if a.subjects:
        keep = set(open(a.subjects).read().split()); cases = [c for c in cases if c[0] in keep]
    idx = rng.permutation(len(cases)); idx = idx[: a.n] if a.n > 0 else idx   # --n 0 = every case (2026-10-13)
    if a.chunk:
        idx = idx[a.chunk[0]::a.chunk[1]]
    # The output channel count comes from the checkpoint. --sr_calibrate arms carry TWO
    # (residual + its amplitude); loading them with out_channels=1 would fail on final_conv, and reading channel 0
    # alone would score the UNCALIBRATED prediction under a calibrated name. Plain arms still have one.
    def _n_out(path):
        ck = torch.load(path, map_location="cpu", weights_only=False); sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
        k = [k for k in sd if k.endswith("final_conv.weight")]; return int(sd[k[0]].shape[0]) if k else 1
    n_out = {p: _n_out(p) for p in a.ckpts}
    def _is_unc(path):   # --sr_uncertainty arms carry a marker buffer; channel 1 is log b, NOT a sigmoid amplitude
        ck = torch.load(path, map_location="cpu", weights_only=False); sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
        return "sr_uncertainty_marker" in sd
    uncertain = {p.rstrip("/").split("/")[-4] + "/" + os.path.basename(p): _is_unc(p) for p in a.ckpts}
    models = {p.rstrip("/").split("/")[-4] + "/" + os.path.basename(p): td.load_task_model(p, dev, out_channels=n_out[p], bn_mode=a.bn_mode) for p in a.ckpts}
    calibrated = {p.rstrip("/").split("/")[-4] + "/" + os.path.basename(p): (n_out[p] == 2 and not _is_unc(p)) for p in a.ckpts}   # BN convention 2026-10-14
    print(f"[sr_eval] {a.dataset}: {len(idx)} scans, thickness {a.thick} mm on axis {a.axis}, bn_mode {a.bn_mode}, models {list(models)}", flush=True)
    rows = []; fh = open(a.out, "w", newline="") if a.out else None; wr = None
    for i in idx:
        sub, t1, gtp = cases[i]; t0 = time.time()
        nii = nib.load(t1); img, _ = to_ras(nii); zooms = nii.header.get_zooms()[:3]
        # 2026-10-20: the degraded axis must be 1 mm (the 3 mm slice model counts voxels); in-plane may sit within 0.1 mm of it (WMH GE3T 3DT1 is
        # 0.94 x 0.94 x 1.0, scored on its native grid, no resampling). ATLAS/BraTS are 1 mm isotropic and unaffected.
        if abs(zooms[a.axis] - 1.0) > 0.05 or max(abs(z - 1.0) for z in zooms) > 0.1: print(f"skip {sub}: not 1 mm ({zooms})", flush=True); continue
        if a.dataset == "ABIDE":
            sp = t1[:-len("T1w.nii")] + "brainseg_with_extracerebral.nii"
            if not os.path.isfile(sp): continue
            lab = LUT[np.clip(np.asanyarray(nib.load(sp).dataobj).squeeze().astype(np.int64), 0, LUT.size - 1)]; lab = to_ras(nib.Nifti1Image(lab.astype(np.int16), nii.affine))[0].astype(np.int64)
            gt = np.zeros(img.shape, bool)
        else:
            lp = os.path.join(droot, a.labels, f"{sub}_seg.nii.gz")
            if not os.path.isfile(lp): continue
            lab = to_ras(nib.load(lp))[0].astype(np.int64); gt = np.isin(to_ras(nib.load(gtp))[0].astype(np.int32), a.gt_label)
            if not gt.any(): continue
        hr = torch.from_numpy(img.astype(np.float32)).to(dev); labt = torch.from_numpy(lab).to(dev); gtt = torch.from_numpy(gt).to(dev)
        brain = (labt > 0) & ~torch.isin(labt, EX.to(dev))
        lr, M, lrc = degrade(hr, a.axis, a.thick, dev, return_scale=True, cubic=True)
        lr_n, (lo, hi) = td.normalize(lr); hr_n = td.apply_scale(hr / M, lo, hi)   # same frame as sr_step: ONE constant for both (see degrade), then the LR's [p1, p99]
        lrc_n = td.apply_scale(lrc, lo, hi)   # the cubic baseline shares the target's frame EXACTLY: same M, same [p1, p99] as the trilinear input.
                                              # Re-normalising it to its own percentiles would re-create the 2026-10-14 frame bug in a new place.
        halo = (F.max_pool3d(gtt.float()[None, None], 11, stride=1, padding=5)[0, 0] > 0) & ~gtt & brain if gt.any() else torch.zeros_like(brain)
        rest = brain & ~gtt & ~halo
        # Lesion boundary (within 2 voxels of the mask edge) vs core (eroded by 2). The recoverable energy
        # of a smooth interior is at its edge; a net can "match cubic in the core and sharpen the boundary" only if we measure both.
        core = (F.max_pool3d((~gtt).float()[None, None], 5, stride=1, padding=2)[0, 0] == 0) & gtt if gt.any() else torch.zeros_like(brain)
        bnd = gtt & ~core
        # --lesion_dilate N scores the lesion PSNR over the mask dilated by N voxels (brain only), so the edge where the
        # thick slices blur the lesion and where super-resolution acts is inside the region; halo/contrast/core/boundary/gate stay on the mask.
        gtl = ((F.max_pool3d(gtt.float()[None, None], 2 * a.lesion_dilate + 1, stride=1, padding=a.lesion_dilate)[0, 0] > 0) & brain) if (a.lesion_dilate > 0 and gt.any()) else gtt
        r = dict(case=sub, lesion_mL=round(float(gtt.sum()) / 1000, 2), psnr_in=round(psnr(lr_n, hr_n, brain), 3),
                 psnr_in_cubic=round(psnr(lrc_n, hr_n, brain), 3))
        if gt.any():
            r.update(psnr_in_lesion=round(psnr(lr_n, hr_n, gtl), 3), psnr_in_rest=round(psnr(lr_n, hr_n, rest), 3),
                     psnr_in_cubic_lesion=round(psnr(lrc_n, hr_n, gtl), 3), psnr_in_cubic_rest=round(psnr(lrc_n, hr_n, rest), 3),
                     contrast_orig=round(float(hr_n[gtt].mean() - hr_n[halo].mean()), 4), contrast_in=round(float(lr_n[gtt].mean() - lr_n[halo].mean()), 4),
                     contrast_in_cubic=round(float(lrc_n[gtt].mean() - lrc_n[halo].mean()), 4))
        base = lrc_n if a.net_input == "cubic" else lr_n   # what the NETWORK sees; the baseline columns above always stay on their own arrays
        # 2026-11-08: visible-component lesion region (see visible_union); nan where fewer than 50 visible voxels
        _hv = False
        if gt.any():
            _vis_np, _nvis = visible_union(gt, brain.cpu().numpy(), base.cpu().numpy()); _vis = torch.from_numpy(_vis_np).to(dev); _hv = int(_vis_np.sum()) >= 50
            r["vis_components"] = int(_nvis); r["vis_vox"] = int(_vis_np.sum())
            r["psnr_in_lesion_vis"] = round(psnr(lr_n, hr_n, _vis), 3) if _hv else float("nan"); r["psnr_in_cubic_lesion_vis"] = round(psnr(lrc_n, hr_n, _vis), 3) if _hv else float("nan")
        # ---- 2026-11-02 lesion-deficit diagnostics, per case ----
        # D2: is the TARGET clamped inside the lesion? hr_n = apply_scale(hr/M, lo, hi) clamps to [lo, hi] of the LR's [p1,p99]; a bright
        # lesion above p99 has its target capped while the prediction is not, which penalises reconstructing it correctly.
        _hf = hr / M; _clip = (_hf > hi) | (_hf < lo)
        r["clip_brain_frac"] = round(float(_clip[brain].float().mean()), 4)
        if gt.any(): r["clip_lesion_frac"] = round(float(_clip[gtt].float().mean()), 4); r["clip_rest_frac"] = round(float(_clip[rest].float().mean()), 4)
        gate_w = None
        if a.lesion_gate == "gt":
            gate_w = gtt.float()                                   # 1 inside the real lesion: residual fully suppressed there (ORACLE)
        elif a.lesion_gate == "detector":
            if not a.gate_deriv: raise SystemExit("--lesion_gate detector needs --gate_deriv")
            _gp = os.path.join(droot, a.gate_deriv, f"{sub}_reg.nii.gz")
            if not os.path.isfile(_gp): raise SystemExit(f"FATAL: no detector map for {sub}: {_gp}")   # fail loudly, never skip
            _gn = nib.load(_gp)
            # 2026-11-08: the map must live on the NATIVE grid. A derivative built from a mis-headed input (RAS data under
            # the native affine, an earlier version of the degraded-tree script) passes the shape test on BraTS -- LPS,
            # flips only -- while being mirrored against the image. Axis codes and shape must match the native file.
            if nib.aff2axcodes(_gn.affine) != nib.aff2axcodes(nii.affine) or _gn.shape != nii.shape:
                raise SystemExit(f"FATAL: gate map {sub} axcodes {nib.aff2axcodes(_gn.affine)} shape {_gn.shape} != native "
                                 f"{nib.aff2axcodes(nii.affine)} {nii.shape}: the map is not on the native grid")
            _st = torch.from_numpy(to_ras(_gn)[0].astype(np.float32)).to(dev)
            if _st.shape != hr.shape: raise SystemExit(f"FATAL: map shape {tuple(_st.shape)} != image {tuple(hr.shape)} for {sub}")
            _dev = (2.0 * _st - 1.0).abs()                          # back to network units: stored = (reg+1)/2
            gate_w = (_dev / a.gate_tau).clamp(0.0, 1.0)
            if gt.any():                                            # how well does the detector's gate cover the real lesion?
                r["gate_mean_lesion"] = round(float(gate_w[gtt].mean()), 4); r["gate_mean_rest"] = round(float(gate_w[rest].mean()), 4)
        for name, m in models.items():
            x = base[None, None]; sh = x.shape[2:]; pads = [(32 - s % 32) % 32 for s in sh]
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                res = m(F.pad(x, (0, pads[2], 0, pads[1], 0, pads[0])))
            _res = res.float()[:, :, :sh[0], :sh[1], :sh[2]]
            _tag0 = name.split("/")[0].replace("super_resolve_", "")   # needed here; redefined identically below
            if uncertain.get(name, False):
                # --sr_uncertainty: channel 0 = residual (applied as is), channel 1 = log b; the scale inside/outside the lesion is written out.
                _b = torch.exp(_res[0, 1].clamp(-7.0, 2.0)); rr = _res[0, 0]
                if gt.any():
                    r[f"unc_b_lesion[{_tag0}]"] = round(float(_b[gtt].mean()), 5); r[f"unc_b_rest[{_tag0}]"] = round(float(_b[rest].mean()), 5)
                if a.lesion_gate == "unc":   # 2026-11-10: the arm's OWN uncertainty as the gate -- no mask, no detector, nothing but the input
                    _wu = (_b / a.gate_tau_unc).clamp(0.0, 1.0); rr = rr * (1.0 - _wu)
                    if gt.any(): r[f"gate_mean_lesion[{_tag0}]"] = round(float(_wu[gtt].mean()), 4); r[f"gate_mean_rest[{_tag0}]"] = round(float(_wu[rest].mean()), 4)
            elif calibrated.get(name, False):
                # channel 0 = residual, channel 1 = its amplitude through a sigmoid (train_downstream.py --sr_calibrate).
                # The applied fraction inside/outside the lesion is the whole point of the arm, so it is written out.
                _amp = torch.sigmoid(_res[0, 1]); rr = _res[0, 0] * _amp
                if gt.any():
                    r[f"amp_mean_lesion[{_tag0}]"] = round(float(_amp[gtt].mean()), 4); r[f"amp_mean_rest[{_tag0}]"] = round(float(_amp[rest].mean()), 4)
            else:
                rr = _res[0, 0]
            if a.residual_norm != "none":   # 2026-10-20: intensity-preserving residual (decided a priori: super-resolution must not change a scan's
                # global intensity relation to its input). offset = zero brain-mean residual; affine = also no global gain: the least-squares
                # affine component of the residual along the INPUT over the brain is removed, so the output's affine fit onto the input is the
                # identity. Uses the input's own statistics only -- no target, no labels -- and is applied identically to every case.
                xb, yb = base[brain], rr[brain]; xm, ym = xb.mean(), yb.mean()
                if a.residual_norm == "offset": rr = rr - ym
                else: slope = ((xb - xm) * (yb - ym)).sum() / ((xb - xm) ** 2).sum().clamp_min(1e-12); rr = rr - (slope * (base - xm) + ym)
            # D1 / D1b: the network's OWN residual inside the real lesion vs the rest, and whether it points the right way.
            # Measured on the ungated residual, so the gate experiments below do not contaminate the diagnosis.
            _tru = hr_n - base                                                     # the correction the net SHOULD apply
            _tag0 = name.split("/")[0].replace("super_resolve_", "")
            # Two checkpoints of the SAME run in one pool (e.g. H@100k and H@200k for a matched-clean-budget
            # read) collapsed onto one column name and the second silently overwrote the first. Append @<step> whenever the pool
            # holds another checkpoint of that run directory; pools of distinct runs keep the exact column names of every pool so far.
            if sum(1 for n2 in models if n2.split("/")[0] == name.split("/")[0]) > 1:
                _tag0 += "@" + os.path.basename(name).replace("super_resolve_", "").replace(".pth", "")
            r[f"res_abs_brain[{_tag0}]"] = round(float(rr[brain].abs().mean()), 4)
            # the same alignment numbers in the REST region: without them "0.28 in the lesion" has no reference (2026-11-02)
            _xr, _yr = rr[rest], _tru[rest]
            r[f"res_sign_rest[{_tag0}]"] = round(float((torch.sign(_xr) == torch.sign(_yr)).float().mean()), 4)
            r[f"res_corr_rest[{_tag0}]"] = round(float(((_xr - _xr.mean()) * (_yr - _yr.mean())).sum() / ((_xr - _xr.mean()).norm() * (_yr - _yr.mean()).norm()).clamp_min(1e-12)), 4)
            if gt.any():
                r[f"res_abs_lesion[{_tag0}]"] = round(float(rr[gtt].abs().mean()), 4)
                r[f"res_abs_rest[{_tag0}]"]   = round(float(rr[rest].abs().mean()), 4)
                r[f"res_sign_lesion[{_tag0}]"] = round(float((torch.sign(rr[gtt]) == torch.sign(_tru[gtt])).float().mean()), 4)   # 0.5 = coin flip
                _x, _y = rr[gtt], _tru[gtt]
                _c = (((_x - _x.mean()) * (_y - _y.mean())).sum() / ((_x - _x.mean()).norm() * (_y - _y.mean()).norm()).clamp_min(1e-12)) if _x.numel() > 8 else float("nan")
                r[f"res_corr_lesion[{_tag0}]"] = round(float(_c), 4)                     # +1 = residual proportional to the true correction (nan: < 9 voxels)
            if gate_w is not None and a.lesion_gate != "unc": rr = rr * (1.0 - gate_w)                                 # suppress the residual where the gate fires
            pred = base + rr
            tag = _tag0 + ("|cal" if calibrated.get(name, False) else "") + ("|unc" if uncertain.get(name, False) else "") + ("" if a.residual_norm == "none" else f"|{a.residual_norm}") + ("|cubicin" if a.net_input == "cubic" else "") \
                  + ("" if a.lesion_gate == "none" else (f"|gate=gt" if a.lesion_gate == "gt" else f"|gate=unc{a.gate_tau_unc:g}" if a.lesion_gate == "unc" else f"|gate=det{a.gate_tau:g}"))
            r[f"psnr_out[{tag}]"] = round(psnr(pred, hr_n, brain), 3)
            if gt.any():
                r[f"psnr_lesion[{tag}]"] = round(psnr(pred, hr_n, gtl), 3); r[f"psnr_rest[{tag}]"] = round(psnr(pred, hr_n, rest), 3)
                r[f"psnr_lesion_vis[{tag}]"] = round(psnr(pred, hr_n, _vis), 3) if _hv else float("nan")   # 2026-11-08: visible components only
                r[f"contrast_out[{tag}]"] = round(float(pred[gtt].mean() - pred[halo].mean()), 4)
                _ret = component_retention(gt, brain.cpu().numpy(), base.cpu().numpy(), pred.cpu().numpy())   # 2026-11-04 retention gate vs the NET INPUT
                r["comp_n"] = _ret["n"]; r["comp_visible"] = _ret["visible"]
                r[f"comp_lost95[{tag}]"] = _ret["lost95"]; r[f"comp_min_ratio[{tag}]"] = _ret["min_ratio"]; r[f"comp_med_ratio[{tag}]"] = _ret["med_ratio"]
                # 2026-11-03: ALWAYS written (nan when the eroded core is empty) -- a conditional key crashed csv.DictWriter on the next case
                _hc = bool(core.any())
                r[f"psnr_core[{tag}]"] = round(psnr(pred, hr_n, core), 3) if _hc else float("nan"); r[f"psnr_in_cubic_core"] = round(psnr(lrc_n, hr_n, core), 3) if _hc else float("nan")
                r[f"psnr_bnd[{tag}]"] = round(psnr(pred, hr_n, bnd), 3); r[f"psnr_in_cubic_bnd"] = round(psnr(lrc_n, hr_n, bnd), 3)
        if a.ext_dir:   # 2026-11-05: an EXTERNAL super-resolution result (dmprior's sr task) scored in the SAME frame and regions as the arms:
            # resampled onto this case's RAS 1 mm grid through the two affines (linear), intensity-matched by least squares to the NET INPUT over
            # the brain (never to the target), then the arms' columns (gate not applied: it is ours). Missing output -> nan columns + ext_note.
            tag = a.ext_tag + ("|cubicin" if a.net_input == "cubic" else "")
            cols = [f"psnr_out[{tag}]", f"psnr_lesion[{tag}]", f"psnr_rest[{tag}]", f"contrast_out[{tag}]", f"psnr_core[{tag}]", f"psnr_bnd[{tag}]",
                    f"comp_lost95[{tag}]", f"comp_min_ratio[{tag}]", f"comp_med_ratio[{tag}]"]
            for c in cols: r[c] = float("nan")
            r[f"ext_match[{tag}]"] = ""; r["ext_note"] = ""
            hits = sorted(glob.glob(os.path.join(a.ext_dir, a.ext_pattern.format(sub=sub))))
            if not hits: r["ext_note"] = "no external output"
            else:
                try:
                    sn = nib.load(hits[0]); sa = np.asarray(sn.dataobj, dtype=np.float32)
                    while sa.ndim > 3: sa = sa[..., 0]
                    ras_aff = nib.as_closest_canonical(nii).affine; T = np.linalg.inv(sn.affine) @ ras_aff
                    co = T[:3, :3] @ np.indices(tuple(hr.shape), dtype=np.float64).reshape(3, -1) + T[:3, 3:4]
                    from scipy import ndimage as _ndi
                    sr = torch.from_numpy(_ndi.map_coordinates(np.ascontiguousarray(sa), co, order=1, mode="constant", cval=0.0).reshape(tuple(hr.shape)).astype(np.float32)).to(dev)
                    gm = bool(np.abs(T[:3] - np.round(T[:3])).max() < 1e-3)
                    xs, ys = sr[brain].double(), base[brain].double(); xm, ym = xs.mean(), ys.mean()
                    aa = float(((xs - xm) * (ys - ym)).sum() / ((xs - xm) ** 2).sum().clamp(min=1e-12)); bb = float(ym) - aa * float(xm); pred = aa * sr + bb
                    r[f"psnr_out[{tag}]"] = round(psnr(pred, hr_n, brain), 3); r[f"ext_match[{tag}]"] = f"{aa:.4g} {bb:.4g} grid{int(gm)}"
                    if gt.any():
                        r[f"psnr_lesion[{tag}]"] = round(psnr(pred, hr_n, gtl), 3); r[f"psnr_rest[{tag}]"] = round(psnr(pred, hr_n, rest), 3)
                        r[f"contrast_out[{tag}]"] = round(float(pred[gtt].mean() - pred[halo].mean()), 4)
                        _ret = component_retention(gt, brain.cpu().numpy(), base.cpu().numpy(), pred.cpu().numpy())
                        r[f"comp_lost95[{tag}]"] = _ret["lost95"]; r[f"comp_min_ratio[{tag}]"] = _ret["min_ratio"]; r[f"comp_med_ratio[{tag}]"] = _ret["med_ratio"]
                        if bool(core.any()): r[f"psnr_core[{tag}]"] = round(psnr(pred, hr_n, core), 3)
                        r[f"psnr_bnd[{tag}]"] = round(psnr(pred, hr_n, bnd), 3)
                except Exception as e: r["ext_note"] = f"unreadable external output: {e}"
        r["sec"] = round(time.time() - t0, 1); r["bn_mode"] = a.bn_mode; rows.append(r)   # trailing column (2026-10-14)
        if fh and wr is None: wr = csv.DictWriter(fh, fieldnames=list(r)); wr.writeheader()
        if wr: wr.writerow(r); fh.flush()
        print(f"{sub:14s} " + " ".join(f"{k} {v}" for k, v in r.items() if k.startswith(("psnr", "contrast"))), flush=True)
    keys = [k for k in rows[0] if k.startswith(("psnr", "contrast", "comp_"))]
    print(f"\n# {a.dataset}, medians over {len(rows)} scans, {a.thick} mm slices on axis {a.axis}")
    for k in keys: print(f"  {k:36s} {np.nanmedian([r[k] for r in rows if k in r]):8.3f}")


if __name__ == "__main__":
    main()
