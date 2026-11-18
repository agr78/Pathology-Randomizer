import os
"""False-positive rate vs z on HELD-OUT healthy scans -- a leak-free operating point.

The z cut cannot be calibrated on the test sets (oracle) and calibrating on
validation_set_v2 fails because its prevalence is 0.16% against 1-6% real
(z* came out at 4.5-5.75, which would make real BraTS FLAIR worse). Nor can z
be chosen per pathology at deployment, since the pathology is unknown.

What IS available: on a genuinely healthy scan the correct flagged fraction
is zero, so flagged% as a function of z is the false-positive rate, measurable
with no lesion labels at all. Setting z at a fixed specificity (a convention,
e.g. 99%) is the standard anomaly-detection operating point: real domain,
pathology-agnostic, no ground truth anywhere.

Uses ADNI3 cognitively-normal subjects. They are excluded from training unless
--adni (HEALTHY_EXTRA_DATASETS is () by default and no sprint set it), so the
model has never seen them -- the 1071 training scans would give an optimistic
FPR. Brain mask comes from the seg head's own output, since no external tissue
derivative exists for ADNI3.

    python eval/fpr_healthy.py --ckpt <pth> --limit 100

--dataset oasis3 scores the OASIS3 CDR-0 release scans listed in the match CSV (OASIS_CSV)
(healthy=1; --modality FLAIR|T1w|T2w picks the file column; 2-D axial FLAIR like the WMH challenge). OASIS3 label maps are
admitted to training only with --oasis, so for every arm without that flag these scans are held out; --subjects restricts to
the reserved held-out list (docs/files/oasis3_heldout_120.txt) so the numbers stay valid for arms trained with --oasis later.
--bn_mode batch = the batch-statistics readout of the Dice legs. --csv writes one row per scan (FPR at every z).
Caveat: CDR-0 elderly FLAIR carries age-related WMH, so this is a healthy-ANATOMY set, not a lesion-free one.
"""
import argparse, csv, glob, os, sys
import numpy as np, torch, nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F
from compare_datasets import (EXTRACEREBRAL_COMPACT_IDX as EX, normalize_like_training,  # noqa: E402
                              to_ras, head_bbox, bn_batch_mode)
from networks import load_checkpoint  # noqa: E402
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from dice_by_mask import apply_postproc  # noqa: E402  (--postproc median:4 = the benchmark's MF rows, 2026-10-13)

def median_filter_device(vol, spec, device):
    """--postproc median:K on the working 1 mm grid; on a GPU the K^3 window median runs in torch (the SciPy version made the
    HCP-Aging jobs CPU-bound at ~3 % GPU use). Same result as dice_by_mask.apply_postproc on 1 mm data."""
    if not spec:
        return vol
    kind, _, k = spec.partition(":"); k = int(round(float(k)))
    if kind != "median" or not str(device).startswith("cuda"):
        return apply_postproc(vol, spec, (1.0, 1.0, 1.0))
    x = torch.as_tensor(vol, device=device); lo = k // 2; hi = k - 1 - lo   # SciPy's even-kernel alignment (verified exact)
    xp = torch.nn.functional.pad(x[None, None], (lo, hi, lo, hi, lo, hi), mode="replicate")[0, 0]
    out = torch.empty_like(x); D = x.shape[0]; step = max(1, int(2e8 // (x.shape[1] * x.shape[2] * k ** 3)))   # chunks of ~200M window values
    for z0 in range(0, D, step):
        z1 = min(D, z0 + step); w = xp[z0:z1 + k - 1].unfold(0, k, 1).unfold(1, k, 1).unfold(2, k, 1)   # (z, y, x, k, k, k)
        out[z0:z1] = torch.kthvalue(w.reshape(*w.shape[:3], -1), k ** 3 // 2 + 1, dim=-1).values   # SciPy's rank for even kernels (verified exact)
    return out.float().cpu().numpy()


ROOT = os.environ.get('PR_EVAL_DATA_ROOT', '/path/to/eval_data_root') + "/ADNI3"
OASIS_CSV = os.environ.get('PR_OASIS_CSV', '/path/to/oasis3_flair_match.csv')
# The OASIS3 scans to use are the BIDS multicontrast tree, not the IDA release tree the match CSV points at.
# Expected layout sub-OAS3xxxx/ses-dNNNN/anat/sub-..._ses-..._<FLAIR|T1w|T2w>.nii.gz; MR_ID OAS30001_MR_d4467 -> sub-OAS30001, ses-d4467.
OASIS_BIDS = os.environ.get("PR_OASIS_BIDS", "/path/to/OASIS_BIDS_Multicontrast")
ZS = np.round(np.arange(0.5, 6.01, 0.25), 2)
# 2026-10-06: the MAD-z scale is unusable for the current head -- healthy |r| is ~0 almost everywhere, so MAD ~ 0 and
# 15-25% of the brain sits above z=3 (OASIS3 run 10169494). The operating points that match the Dice legs are FIXED
# |r| thresholds (the global oracle threshold of each lesion dataset is one of them), so FPR is also reported on this
# grid and the full per-scan histogram of |r| over the brain is saved (--hist), so any threshold can be read later.
THR = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
HBINS = np.linspace(0.0, 1.5, 301)   # 0.005-wide bins; the head is unbounded but |r| > 1.5 is not seen on healthy tissue
# 2026-10-07: connect the false positives to the anatomy. Per pooled class (anatomy head's own labels, compact scheme) the flagged
# fraction at each threshold; a bit-packed flag mask (bit0 |r|>0.1, bit1 >0.2, bit2 >0.3) saved per scan so flagged voxels can be
# split post hoc into 'both segmenters agree on the anatomy' (a false positive proper) vs 'they disagree' (ambiguous or truly changed
# anatomy, e.g. age-related WMH) against the healthy-only SynthSeg's segmentation of the same scan. A seg-only checkpoint (that
# network) yields the tissue-inconsistency RULE map instead (tanh(s/4), s = (x - med_c)/(1.4826 MAD_c), L/R WM pooled), so the
# SynthSeg + rule baseline gets the same FPR columns on the same scans.
POOLS = {"wm": (20, 38), "cortex": (21, 39), "deepgm": (26, 27, 28, 29, 44, 45, 46, 47), "vent": (22, 23, 40, 41), "scsf": (4,),
         "cereb": (24, 25, 42, 43), "bstem": (30,)}
RULE_KAPPA = 4.0


def rule_map(xw, seg_idx, brain):
    """tissue-inconsistency rule on the working-grid image xw (normalized [0,1]) and the argmax seg: signed tanh(s/kappa)."""
    seg_eff = np.where(np.isin(seg_idx, (20, 38)), 20, seg_idx); out = np.zeros_like(xw)
    for c in np.unique(seg_eff[brain]):
        if int(c) in EX or c == 0: continue
        m = (seg_eff == c) & brain
        if m.sum() < 200: continue
        med = np.median(xw[m]); mad = 1.4826 * np.median(np.abs(xw[m] - med)) + 1e-6
        out[m] = np.tanh((xw[m] - med) / mad / RULE_KAPPA)
    return out
CN = {"cn", "control", "normal", "hc", "0", "0.0", "healthy"}


def healthy_ids(tsv):
    ids = set()
    with open(tsv) as f:
        r = csv.DictReader(f, delimiter="\t")
        idc = next((c for c in r.fieldnames if "participant" in c.lower() or c.lower() in ("id", "subject")), None)
        dxc = next((c for c in r.fieldnames if any(k in c.lower() for k in ("dx", "diag", "group"))), None)
        print(f"participants.tsv columns: {r.fieldnames}   using id={idc} dx={dxc}")
        if not (idc and dxc):
            return ids
        for row in r:
            dx = str(row[dxc]).lower().strip()
            if dx in CN or "cognitively normal" in dx:
                ids.add(str(row[idc]).strip().replace("sub-", ""))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset", default="adni3", choices=("adni3", "oasis3", "oasis3_bids", "glob"))
    ap.add_argument("--img_glob", default=None, help="--dataset glob: recursive file glob of the scans to score (e.g. HCP-Aging: '/path/to/hcp_aging/**/*T1w*.nii*'); --subjects (optional) keeps files whose path contains one of the listed ids")
    ap.add_argument("--modality", default="T1w", help="oasis3: FLAIR | T1w | T2w (column of the match CSV)")
    ap.add_argument("--subjects", default=None, help="oasis3: file with one subj_ID per line (the reserved held-out list)")
    ap.add_argument("--bn_mode", default="running", choices=("running", "batch"))
    ap.add_argument("--csv", default=None, help="per-scan output: subject, modality, n_brain, FPR at every z and every fixed threshold")
    ap.add_argument("--postproc", default="", help="median:K (mm) applied to the SIGNED map on the 1 mm working grid before |r| (the benchmark reads healthy FPR on median-filtered maps); default none")
    ap.add_argument("--hist", default=None, help="npz with the per-scan histogram of |r| over the brain (bins HBINS), for any later threshold")
    a = ap.parse_args()

    if a.dataset == "glob":
        assert a.img_glob, "--dataset glob needs --img_glob"
        files = sorted(glob.glob(a.img_glob, recursive=True))
        if a.subjects:
            keep = [ln.split()[0] for ln in open(a.subjects) if ln.strip() and not ln.startswith("#")]
            files = [f for f in files if any(k in f for k in keep)]
        print(f"glob {a.modality}: {len(files)} scans match {a.img_glob}; using {min(a.limit, len(files))}")
    elif a.dataset == "adni3":
        ids = healthy_ids(os.path.join(ROOT, "participants.tsv"))
        files = sorted(glob.glob(os.path.join(ROOT, "sub-*", "**", "*T1w*.nii.gz"), recursive=True))
        files = [f for f in files if any(sid in f for sid in ids)]
        print(f"ADNI3: {len(ids)} CN ids, {len(files)} CN T1w files found; using {min(a.limit, len(files))}")
    elif a.dataset == "oasis3_bids":
        rows = [ln.split() for ln in open(a.subjects) if ln.strip() and not ln.startswith("#")] if a.subjects else None
        if rows is None:
            with open(OASIS_CSV) as fh:
                rows = [(r["subj_ID"], r["MR_ID"]) for r in csv.DictReader(fh) if r["healthy"] == "1"]
        files = []; missing = []
        for subj, mrid in rows:
            oas, day = mrid.split("_MR_")                       # OAS30001_MR_d4467 -> sub-OAS30001 / ses-d4467
            # files are .nii (some .nii.gz) and may carry acq-/run- tags: prefer the untagged scan, then run-01, then anything
            pats = [os.path.join(OASIS_BIDS, f"sub-{oas}", f"ses-{day}", "anat", f"*_{a.modality}.nii*"),
                    os.path.join(OASIS_BIDS, f"sub-{oas}", "**", f"*ses-{day}*_{a.modality}.nii*")]
            hit = next((glob.glob(p, recursive=True) for p in pats if glob.glob(p, recursive=True)), [])
            hit = sorted(hit, key=lambda q: ("acq-" in os.path.basename(q), "run-" in os.path.basename(q), os.path.basename(q)))
            (files.append(hit[0]) if hit else missing.append(mrid))
        print(f"OASIS3 BIDS {a.modality}: {len(files)} scans found, {len(missing)} sessions without one" + (f" (e.g. {missing[:3]})" if missing else "") + f"; using {min(a.limit, len(files))}")
    else:
        keep = None
        if a.subjects:
            keep = {ln.split()[0] for ln in open(a.subjects) if ln.strip() and not ln.startswith("#")}
        col = f"{a.modality}_file"
        with open(OASIS_CSV) as fh:
            rows = [r for r in csv.DictReader(fh) if r["healthy"] == "1" and r.get(col) and (keep is None or r["subj_ID"] in keep)]
        rows.sort(key=lambda r: r["subj_ID"]); files = [r[col] for r in rows]
        print(f"OASIS3 CDR-0 {a.modality}: {len(files)} scans" + (f" (held-out list {a.subjects})" if a.subjects else "") + f"; using {min(a.limit, len(files))}")
    if not files:
        print("NO FILES -- check the glob / CSV columns"); return

    model, _, _ = load_checkpoint(a.ckpt, a.device, verbose=False); model.eval()
    if a.bn_mode == "batch":
        print(f"[bn_mode] batch statistics in {bn_batch_mode(model)} BatchNorm layers")
    flag = {z: [] for z in ZS}; flag_t = {t: [] for t in THR}; n = 0; per_scan = []; hists = []; meds = []; mads = []
    seg_dir = flag_dir = None
    if a.csv:
        seg_dir = os.path.splitext(os.path.abspath(a.csv))[0] + "_segs"; os.makedirs(seg_dir, exist_ok=True)
        flag_dir = os.path.splitext(os.path.abspath(a.csv))[0] + "_flags"; os.makedirs(flag_dir, exist_ok=True)
    with torch.no_grad():
        for f in files[: a.limit]:
            # Same pipeline as compare_datasets.run_inference: RAS, head-crop,
            # normalise, resample to 1 mm, PAD TO A MULTIPLE OF 32 (the U-Net
            # has 4 pooling levels; an odd shape breaks the skip connections,
            # which is what crashed the first attempt: "Expected size 48 but
            # got size 49"). Score only the unpadded region.
            nii = nib.load(f)
            if len(nii.shape) == 4:   # IDA release volumes can carry a trailing singleton
                nii = nib.Nifti1Image(np.asanyarray(nii.dataobj)[..., 0], nii.affine, nii.header)
            img, _ = to_ras(nii)
            zooms = np.array(nii.header.get_zooms()[:3], dtype=np.float32)
            work = img[head_bbox(img)].astype(np.float32)
            x = normalize_like_training(torch.from_numpy(work).to(a.device))[None, None]
            if np.any(np.abs(zooms - 1.0) > 0.05):
                new_shape = tuple(int(max(round(s * z), 1)) for s, z in zip(x.shape[2:], zooms))
                x = F.interpolate(x, size=new_shape, mode="trilinear", align_corners=False)
            shp = x.shape[2:]
            pads = [(32 - s % 32) % 32 for s in shp]
            x = F.pad(x, (0, pads[2], 0, pads[1], 0, pads[0]))
            with torch.amp.autocast("cuda"):
                out = model(x)
            seg, reg = (out if isinstance(out, (tuple, list)) else (out, None))
            sl = (slice(0, shp[0]), slice(0, shp[1]), slice(0, shp[2]))
            seg_idx = seg.float()[0].argmax(0).cpu().numpy()[sl]
            if seg_dir:   # 2026-10-06: anatomy on the 1 mm working grid, uint8; same preprocessing for every model -> Dice across runs by filename
                stem = os.path.basename(f).replace(".nii.gz", "").replace(".nii", "")
                if stem in ("T1w", "T2w", "FLAIR", "T1", "T2", "3DT1"):   # generic file names (HCP-Aging, WMH): fold the subject directory into the name (2026-10-13)
                    stem = os.path.basename(os.path.dirname(f)) + "_" + stem
                nib.save(nib.Nifti1Image(seg_idx.astype(np.uint8), np.eye(4)), os.path.join(seg_dir, f"{stem}_seg.nii.gz"))
            brain = (seg_idx > 0) & ~np.isin(seg_idx, list(EX))
            if brain.sum() < 1000:
                continue
            if reg is None:   # seg-only checkpoint (the healthy-only SynthSeg): score the RULE map built from its own segmentation
                xw = x.float()[0, 0].cpu().numpy()[sl]
                r = np.abs(median_filter_device(rule_map(xw, seg_idx, brain), a.postproc, a.device)); map_kind = "rule"
            else:
                r = np.abs(median_filter_device(reg.float()[0, 0].cpu().numpy()[sl], a.postproc, a.device)); map_kind = "head"
            ab = r[brain]
            med = np.median(ab); mad = np.median(np.abs(ab - med)) * 1.4826 + 1e-6
            fz = {z: float(((r > med + z * mad) & brain)[brain].mean()) for z in ZS}
            ft = {t: float((ab > t).mean()) for t in THR}
            fcls = {}
            for pname, ids in POOLS.items():
                pm = np.isin(seg_idx, ids) & brain
                for t in (0.1, 0.2, 0.3):
                    fcls[f"fpr_t{t:g}_{pname}"] = float((r[pm] > t).mean()) if pm.sum() else float("nan")
            if flag_dir:
                fl = ((r > 0.1).astype(np.uint8) | ((r > 0.2).astype(np.uint8) << 1) | ((r > 0.3).astype(np.uint8) << 2)) * brain.astype(np.uint8)
                fstem = os.path.basename(f).replace('.nii.gz', '').replace('.nii', ''); fstem = (os.path.basename(os.path.dirname(f)) + "_" + fstem) if fstem in ("T1w", "T2w", "FLAIR", "T1", "T2", "3DT1") else fstem
                nib.save(nib.Nifti1Image(fl, np.eye(4)), os.path.join(flag_dir, f"{fstem}_flags.nii.gz"))
            for z in ZS:
                flag[z].append(fz[z])
            for t in THR:
                flag_t[t].append(ft[t])
            meds.append(float(med)); mads.append(float(mad)); hists.append(np.histogram(ab, bins=HBINS)[0].astype(np.int64))
            per_scan.append(dict(scan=os.path.basename(f), modality=a.modality, map=map_kind, n_brain=int(brain.sum()), med_abs_r=float(med), mad_abs_r=float(mad),
                                 **{f"fpr_z{z:g}": fz[z] for z in ZS}, **{f"fpr_t{t:g}": ft[t] for t in THR}, **fcls))
            n += 1
    print(f"\n# {os.path.basename(os.path.dirname(os.path.dirname(a.ckpt)))}  n={n} held-out healthy scans" + (f"; segmentations in {seg_dir}" if seg_dir else ""))
    if not meds:
        print("no scans scored"); return
    print(f"map: {per_scan[0]['map'] if per_scan else '?'}  (head = regression output; rule = tissue-inconsistency rule on this network's own segmentation)")
    print(f"healthy |r| in brain: median of per-scan medians {np.median(meds):.4f}, of per-scan MADs {np.median(mads):.4f}  (MAD-z is meaningless when these are ~0)")
    print(f"{'thr |r|':>8} {'FPR% med':>9} {'FPR% p90':>9}")
    for t in THR:
        print(f"{t:8.2f} {np.median(flag_t[t])*100:9.3f} {np.percentile(flag_t[t],90)*100:9.3f}")
    print(f"{'z':>5} {'FPR% med':>9} {'FPR% p90':>9}")
    for z in ZS:
        if z * 4 % 2 == 0:
            print(f"{z:5.2f} {np.median(flag[z])*100:9.3f} {np.percentile(flag[z],90)*100:9.3f}")
    for spec in (0.98, 0.99, 0.995):
        zz = [z for z in ZS if np.median(flag[z]) <= 1 - spec]
        print(f"Z_AT_SPEC {spec:.3f} = {min(zz) if zz else 'none'}   (smallest z with median FPR <= {(1-spec)*100:.1f}%)")
    if a.hist and hists:
        os.makedirs(os.path.dirname(os.path.abspath(a.hist)), exist_ok=True)
        np.savez_compressed(a.hist, bins=HBINS, hist=np.stack(hists), scans=np.array([d["scan"] for d in per_scan]), n_brain=np.array([d["n_brain"] for d in per_scan]))
        print(f"per-scan |r| histograms: {a.hist}")
    if a.csv and per_scan:
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_scan[0])); w.writeheader(); w.writerows(per_scan)
        print(f"per-scan CSV: {a.csv}")


if __name__ == "__main__":
    main()
