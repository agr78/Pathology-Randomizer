#!/usr/bin/env python
"""Manifold enclosure of real lesion components by generator draws (2026-10-13), in the Zernike descriptor space of
eval/manifold_descriptors.py, with the k-NN precision / recall of Kynkaanniemi et al. 2019: a real component is ENCLOSED
(recall) if it lies within the k-NN ball of at least one synthetic component; a synthetic one is PRECISE if it lies within the
k-NN ball of a real component. Descriptors are z-scored on the synthetic set; --with_volume appends log10 volume.
    python eval/manifold_knn.py --synthetic synthetic.npz --real real_atlas_o.npz real_brats.npz real_wmh.npz real_pres.npz --k 5 --fig manifold.png
"""
import argparse, os
import numpy as np
from scipy.spatial import cKDTree


def knn_radius(X, k):
    d, _ = cKDTree(X).query(X, k=k + 1); return d[:, -1]


def enclosure(ref, q, k):
    """fraction of q points inside the union of k-NN balls of ref points."""
    tree = cKDTree(ref); r = knn_radius(ref, k); d, j = tree.query(q, k=min(k * 4, len(ref)))
    return float(np.mean(np.any(d <= r[j], axis=1)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--synthetic", required=True); ap.add_argument("--real", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--with_volume", action="store_true"); ap.add_argument("--fig", default=None); ap.add_argument("--max_synth", type=int, default=20000)
    a = ap.parse_args(); rng = np.random.default_rng(0)
    S = np.load(a.synthetic, allow_pickle=True); Xs = S["desc"]; vs = S["logvol"]
    if a.with_volume: Xs = np.column_stack([Xs, vs])
    if len(Xs) > a.max_synth: idx = rng.choice(len(Xs), a.max_synth, replace=False); Xs = Xs[idx]
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-9; Zs = (Xs - mu) / sd
    print(f"# synthetic: {len(Zs)} components, {Zs.shape[1]} dims (volume {'in' if a.with_volume else 'out'}), k = {a.k}")
    rows = []
    for rp in a.real:
        R = np.load(rp, allow_pickle=True); Xr = R["desc"]; vr = R["logvol"]
        if a.with_volume: Xr = np.column_stack([Xr, vr])
        Zr = (Xr - mu) / sd
        rec = enclosure(Zs, Zr, a.k); prec = enclosure(Zr, Zs, a.k)
        nn = cKDTree(Zs).query(Zr, k=1)[0]; rows.append((os.path.basename(rp), len(Zr), rec, prec, np.median(nn)))
        print(f"{os.path.basename(rp):22s} n={len(Zr):6d}  recall (real enclosed by synthetic) {rec:.3f}   precision (synthetic enclosed by real) {prec:.3f}   median NN dist {np.median(nn):.2f}")
    if a.fig:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from numpy.linalg import svd
        Zc = Zs - Zs.mean(0); U, s_, Vt = svd(Zc, full_matrices=False); P = Vt[:2]
        fig, axes = plt.subplots(1, len(a.real), figsize=(3.2 * len(a.real), 3.2), squeeze=False)
        for ax, rp, (name, n, rec, prec, _) in zip(axes[0], a.real, rows):
            R = np.load(rp, allow_pickle=True); Xr = R["desc"]; Xr = np.column_stack([Xr, R["logvol"]]) if a.with_volume else Xr; Zr = (Xr - mu) / sd
            ax.scatter(*(Zc @ P.T).T, s=2, c="#9ecae1", alpha=0.4, label="Generator draws"); ax.scatter(*((Zr - Zs.mean(0)) @ P.T).T, s=4, c="#d62728", alpha=0.6, label="Real lesions")
            ax.set_title(f"{name.replace('real_', '').replace('.npz', '').upper()}: recall {rec:.2f}", fontsize=9); ax.set_xlabel("PC 1", fontsize=8); ax.set_ylabel("PC 2", fontsize=8); ax.tick_params(labelsize=7)
        axes[0, 0].legend(fontsize=7, loc="upper right"); plt.tight_layout(); fig.savefig(a.fig, dpi=200, bbox_inches="tight"); print("->", a.fig)


if __name__ == "__main__":
    main()
