"""Healthy-SynthSeg control for the zero-shot tables (2026-10-06): no pathology, no regression head. The 500k healthy-only network
(healthy_98ea36c) segments the scan; each voxel is then scored against the intensity statistics of ITS OWN predicted class in the
same scan -- the tissue-inconsistency rule of eval/tissue_inconsistency.py, written out as maps in the project's convention so
dice_by_mask scores it like every other arm (global oracle threshold, median:4mm, synthstrip_b0 brain mask).

    s(v) = (x(v) - med_c) / (1.4826 MAD_c),  c = seg(v), medians over the brain of the same scan, x in the network's [p1,p99] frame
    map  = tanh(s / kappa), stored as (clip(map,-1,1)+1)/2  (signed: brighter than its class > 0.5)

L/R white matter are pooled (as in the network's own channel); classes with < 200 voxels are left at 0; the healthy model has no
lesion class. kappa only sets where tanh saturates (default 4: z = 4 -> 0.76, z = 8 -> 0.96); the oracle threshold is scale-free.
Needs image + healthy seg only (no GT), so it runs wherever the evaluation data and derivatives are mounted.

    python eval/tissue_rule_maps.py --dataset WMH_seg_challenge --modality FLAIR
    python eval/tissue_rule_maps.py --dataset BraTS2023-GLI --modality T2w
    # same rule on OUR network's anatomy head (recipe @500k, batch readout): swaps only the regression head for the rule
    python eval/tissue_rule_maps.py --dataset WMH_seg_challenge --modality FLAIR \
        --seg_run synthseg_pathology_healthy_fork_joint_imtl_att_darkcsf_s1_4e2dd3f_synthseg_500000_bnbatch --out_run synthseg_rule_recipe500k_s1
"""
import argparse, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, nibabel as nib, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compare_datasets import normalize_like_training, find_t1_files, output_name_for, EXTRACEREBRAL_COMPACT_IDX, LABEL_LIST_SEGMENTATION  # noqa: E402

SPACE = os.environ.get('PR_DERIV_ROOT', '/path/to/deriv_root'); RAW = os.environ.get('PR_EVAL_DATA_ROOT', '/path/to/eval_data_root')
WM_IDS = [LABEL_LIST_SEGMENTATION.index(i) for i in (1, 18)]
# compact indices 20..37 are the LEFT structures and 38..55 the matching RIGHT ones in the same order (lut/pathology_compact.lut):
# WM, cortex, lateral + inferior ventricle, cerebellar WM + cortex, thalamus, caudate, putamen, pallidum, hippocampus, amygdala,
# accumbens, hypothalamus, fornix, mammillary body, basal forebrain, septal nuclei. --pool bilateral maps each right index onto its left.
BILATERAL = {38 + i: 20 + i for i in range(18)}
POOL = "wm"
EXCL = set([0] + list(EXTRACEREBRAL_COMPACT_IDX))


def one(args):
    img_p, seg_p, mask_p, out_p, kappa = args
    if os.path.isfile(out_p):
        return "exists"
    if not os.path.isfile(seg_p):
        return "no seg"
    im = nib.load(img_p); img = im.get_fdata().astype(np.float32); seg = np.rint(nib.load(seg_p).get_fdata()).astype(np.int32)
    if seg.shape != img.shape:
        return "shape mismatch"
    brain = nib.load(mask_p).get_fdata() > 0 if os.path.isfile(mask_p) else ((seg > 0) & ~np.isin(seg, list(EXTRACEREBRAL_COMPACT_IDX)))
    x = normalize_like_training(torch.from_numpy(img)).numpy()
    if POOL == "bilateral":
        lut = np.arange(max(int(seg.max()) + 1, 56)); lut[list(BILATERAL)] = list(BILATERAL.values()); seg_eff = lut[seg]
    else:
        seg_eff = np.where(np.isin(seg, WM_IDS), WM_IDS[0], seg)
    s = np.zeros_like(x)
    for c in np.unique(seg_eff[brain]):
        if int(c) in EXCL:
            continue
        m = (seg_eff == c) & brain
        if m.sum() < 200:
            continue
        med = np.median(x[m]); mad = 1.4826 * np.median(np.abs(x[m] - med)) + 1e-6
        s[m] = (x[m] - med) / mad
    out = (np.clip(np.tanh(s / kappa), -1, 1) + 1) / 2
    out[~brain] = 0.5
    hdr = im.header.copy(); hdr.set_data_dtype(np.float32); hdr["scl_slope"] = 1.0; hdr["scl_inter"] = 0.0
    nib.save(nib.Nifti1Image(out.astype(np.float32), im.affine, hdr), out_p)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("WMH_seg_challenge", "ATLAS", "BraTS2023-GLI"))
    ap.add_argument("--modality", default=None, help="WMH: FLAIR | T1;  BraTS: T1w | FLAIR | T2w;  ATLAS: T1w")
    ap.add_argument("--seg_run", default="synthseg_healthy_98ea36c_synthseg_500000", help="derivative stem of the healthy segmentation (WMH gets _<modality>)")
    ap.add_argument("--out_run", default="synthseg_rule_healthy_98ea36c_500000", help="derivative stem to write (WMH gets _<modality>)")
    ap.add_argument("--kappa", type=float, default=4.0); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pool", default="wm", choices=("wm", "bilateral"), help="class statistics pooled over L/R white matter only (as the network channel) or over every bilateral pair (2026-10-09)")
    a = ap.parse_args()
    global POOL; POOL = a.pool
    wmh = a.dataset.startswith("WMH"); data_dir = f"{RAW}/{a.dataset}"; D = f"{SPACE}/{a.dataset}/derivatives"
    mod = a.modality or ("FLAIR" if wmh else "T1w")
    files = find_t1_files(data_dir, a.dataset, modality=mod if wmh else "3DT1")
    if not wmh and mod != "T1w":
        files = [p.replace("_T1w.nii.gz", f"_{mod}.nii.gz") for p in files]; files = [p for p in files if os.path.isfile(p)]
    def with_mod(run):   # WMH folds the modality into the directory name, BEFORE a _bnbatch readout tag if there is one
        if not wmh: return run
        return run.replace("_bnbatch", f"_{mod}_bnbatch") if run.endswith("_bnbatch") else f"{run}_{mod}"
    seg_dir = f"{D}/{with_mod(a.seg_run)}"; out_dir = f"{D}/{with_mod(a.out_run)}"
    os.makedirs(out_dir, exist_ok=True)
    jobs = []
    for p in (files[:a.limit] if a.limit else files):
        name = output_name_for(p, data_dir, a.dataset, mod if wmh else mod)      # <stem>_seg.nii.gz, modality folded in for WMH and non-T1w BraTS
        stem = name[:-len("_seg.nii.gz")]
        jobs.append((p, f"{seg_dir}/{name}", f"{D}/synthstrip_b0/{name}", f"{out_dir}/{stem}_reg.nii.gz", a.kappa))
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, jobs))
    from collections import Counter
    print(f"{a.dataset} {mod}: {len(jobs)} cases -> {out_dir}  {dict(Counter(res))}")


if __name__ == "__main__":
    main()
