"""Healthy-cohort flagged fraction at given |r| cutoffs, from the per-scan histograms fpr_healthy.py saves (2026-10-08).

    python eval/fpr_at_threshold.py --hist <run>_<MOD>_hist.npz --thr 0.218 [--thr 0.395 ...] [--json diceestimate_<arm>.json --modality T1w]
Prints, per cutoff: mean and median over scans of the fraction of brain voxels >= cutoff, the 90th percentile scan, the share of scans above
10 %, and the pooled-voxel fraction. The histogram bins are 0.005 wide; the cutoff snaps to the nearest bin edge (reported)."""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--hist", required=True, nargs="+"); ap.add_argument("--thr", type=float, nargs="*", default=[])
ap.add_argument("--json", default=None, help="thresholds json from diceestimate_threshold.py"); ap.add_argument("--modality", default=None)
a = ap.parse_args()
thrs = list(a.thr)
if a.json:
    j = json.load(open(a.json)); mod = a.modality or next(iter(j["thresholds"]))
    thrs.append(j["thresholds"][mod]["thr"]); print(f"# DiceEstimate threshold for {mod} from {a.json}: {thrs[-1]:.4f}")
for h in a.hist:
    z = np.load(h); bins, hist, nb = z["bins"], z["hist"], z["n_brain"]
    print(f"\n{h}: {len(nb)} scans")
    print(f"{'cutoff':>7s} {'edge':>6s} | {'mean %':>7s} {'median %':>9s} {'p90 %':>7s} {'scans>10%':>10s} | {'pooled %':>9s}")
    for t in thrs:
        k = int(np.argmin(np.abs(bins - t))); fpr = hist[:, k:].sum(1) / nb; pooled = hist[:, k:].sum() / nb.sum()
        print(f"{t:7.3f} {bins[k]:6.3f} | {fpr.mean()*100:7.2f} {np.median(fpr)*100:9.2f} {np.percentile(fpr, 90)*100:7.2f} {(fpr > 0.10).mean()*100:9.1f}% | {pooled*100:9.2f}")
