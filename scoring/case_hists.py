#!/usr/bin/env python
"""Per-case |r| histograms of one lesion leg, saved once (2026-10-13), so that any threshold rule -- the benchmark's DiceEstimate over
many random 10 % validation splits, the oracle, a fixed cutoff -- can be evaluated later without touching the maps again.
Per case: histogram of |r| over lesion voxels and over healthy brain voxels (inside synthstrip_b0 minus the extracerebral classes),
on the RAW signed map and on the 4 mm median-filtered map ("Threshold selection has been performed on the original anomaly maps and
has been transferred to the median-filtered ones", arXiv 2512.01534). NB = 2000 bins over [0, 1].
    python eval/case_hists.py --dataset ATLAS --derivs <arm>_bnbatch --out docs/files/case_hists/atlas_<arm>.npz
    python eval/case_hists.py --dataset BraTS2023-GLI --modality T2w --target 1 2 3 --derivs <arm>_bnbatch --out ...
    python eval/case_hists.py --dataset WMH_seg_challenge --list_modality FLAIR --modality FLAIR --derivs <arm>_FLAIR_bnbatch --out ...
Output npz: cases (str), pos_raw, neg_raw, pos_mf, neg_mf (int64 [n, NB]), n_pos, n_brain, zooms.
"""
import argparse, os, sys
import numpy as np, nibabel as nib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from compare_datasets import EXTRACEREBRAL_COMPACT_IDX, dataset_key, list_cases, output_name_for  # noqa: E402
from dice_by_mask import apply_postproc  # noqa: E402
EX = list(EXTRACEREBRAL_COMPACT_IDX); NB = 2000
_CTX = {}


def one(case):
    a, droot, data_dir, wanted = _CTX["a"], _CTX["droot"], _CTX["data_dir"], _CTX["wanted"]
    sub, img_p, gtp = case[0], case[1], case[2]
    name = output_name_for(img_p, data_dir, a.dataset, a.modality); mask_name = output_name_for(case[3], data_dir, a.dataset, "3DT1") if len(case) > 3 else name
    rp = os.path.join(droot, a.derivs, name.replace("_seg.nii.gz", "_reg.nii.gz")); mp = os.path.join(droot, "synthstrip_b0", mask_name)
    if not (os.path.isfile(gtp) and os.path.isfile(rp) and os.path.isfile(mp)):
        return None
    g_n = nib.load(gtp); gt = np.isin(np.rint(g_n.get_fdata()).astype(np.int32), wanted); zooms = g_n.header.get_zooms()[:3]
    m = nib.load(mp).get_fdata().astype(np.int32); brain = (m > 0) & ~np.isin(m, EX)
    if brain.shape != gt.shape or brain.sum() < 1000 or not (gt & brain).any():
        return None
    rn = nib.load(rp); sreg = rn.get_fdata().astype(np.float32) * 2.0 - 1.0
    out = dict(case=sub, n_pos=int((gt & brain).sum()), n_brain=int(brain.sum()), zooms=np.array(zooms, np.float32))
    for tag, spec in (("raw", ""), ("mf", a.postproc)):
        r = np.abs(apply_postproc(sreg, spec, rn.header.get_zooms()))
        hp = np.bincount(np.clip((r[brain & gt] * (NB - 1)).astype(np.int64), 0, NB - 1), minlength=NB)
        hn = np.bincount(np.clip((r[brain & ~gt] * (NB - 1)).astype(np.int64), 0, NB - 1), minlength=NB)
        out[f"pos_{tag}"], out[f"neg_{tag}"] = hp, hn
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--derivs", required=True); ap.add_argument("--modality", default="3DT1"); ap.add_argument("--list_modality", default=None)
    ap.add_argument("--target", type=int, nargs="+", default=None); ap.add_argument("--postproc", default="median:4"); ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data_dir = os.environ.get('PR_EVAL_DATA_ROOT', '/path/to/eval_data_root') + f"/{a.dataset}"; droot = os.environ.get('PR_DERIV_ROOT', '/path/to/deriv_root') + f"/{a.dataset}/derivatives"
    key = dataset_key(a.dataset); wanted = a.target if a.target else ([1, 2, 3] if key == "BraTS" else [1])
    cases = list_cases(data_dir, a.dataset, modality=a.list_modality)
    if not a.dataset.startswith("WMH") and a.modality not in (None, "3DT1", "T1", "T1w"):
        cases = [(s, t.replace("_T1w.nii.gz", f"_{a.modality}.nii.gz"), g, t) for s, t, g in cases]; cases = [c for c in cases if os.path.isfile(c[1])]
    if a.subjects:
        cases = [c for c in cases if c[0] in a.subjects]
    if a.limit:
        cases = cases[: a.limit]
    _CTX.update(a=a, droot=droot, data_dir=data_dir, wanted=wanted)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.workers) as pool:
        recs = [r for r in pool.imap(one, cases, chunksize=1) if r is not None]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, cases=np.array([r["case"] for r in recs]), n_pos=np.array([r["n_pos"] for r in recs]), n_brain=np.array([r["n_brain"] for r in recs]),
                        zooms=np.stack([r["zooms"] for r in recs]), **{k: np.stack([r[k] for r in recs]) for k in ("pos_raw", "neg_raw", "pos_mf", "neg_mf")},
                        dataset=a.dataset, modality=a.modality, derivs=a.derivs, postproc=a.postproc, nb=NB)
    print(f"# {a.dataset} {a.modality} {a.derivs}: {len(recs)} cases with lesion -> {a.out}")


if __name__ == "__main__":
    main()
