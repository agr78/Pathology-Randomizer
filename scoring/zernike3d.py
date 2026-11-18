#!/usr/bin/env python
"""3-D Zernike descriptors of a binary shape (Novotni & Klein 2003; Canterakis radial polynomials), 2026-10-13.
A component is centred at its centre of mass, scaled into the unit ball (max radius = 1), and the moments
    Omega_nlm = (3 / 4 pi) sum_v f(v) conj(Z_nlm(x_v)) dV,   Z_nlm(r, theta, phi) = R_nl(r) Y_lm(theta, phi)
are summed over its voxels (physical coordinates, anisotropic voxels allowed). The rotation-invariant descriptor is
F_nl = || Omega_nl ||_2 over m, for 0 <= n <= order, l <= n, n - l even. Large components are subsampled to <= max_vox voxels.
Self-test: python eval/zernike3d.py  (90-degree grid rotations must leave F_nl unchanged; a ball has l = 0 moments only).
"""
import numpy as np
from math import comb, sqrt, pi
from scipy.special import sph_harm


def _q(k, l, nu):
    return ((-1) ** (k + nu)) / 4 ** k * sqrt((2 * l + 4 * k + 3) / 3) * comb(2 * k, k) * comb(k, nu) * comb(2 * (k + l + nu) + 1, 2 * k) / comb(k + l + nu, k)


def radial(n, l, r):
    k = (n - l) // 2
    return sum(_q(k, l, nu) * r ** (2 * nu + l) for nu in range(k + 1))


def nl_pairs(order):
    return [(n, l) for n in range(order + 1) for l in range(n + 1) if (n - l) % 2 == 0]


def descriptors(coords, weights, order=10):
    """coords: (N, 3) physical coordinates of the voxels of one component; weights: (N,) voxel volumes. Returns the vector of F_nl."""
    x = coords - (coords * weights[:, None]).sum(0) / weights.sum()
    r = np.linalg.norm(x, axis=1); rmax = r.max() + 1e-9; x = x / rmax; r = r / rmax
    theta = np.arccos(np.clip(x[:, 2] / np.maximum(r, 1e-12), -1, 1)); phi = np.arctan2(x[:, 1], x[:, 0])
    w = weights / weights.sum()   # volume-normalised: the descriptor is scale-free (volume is reported separately)
    out = []
    for n, l in nl_pairs(order):
        R = radial(n, l, r) * w
        om = np.array([np.sum(R * np.conj(sph_harm(m, l, phi, theta))) for m in range(-l, l + 1)]) * (3 / (4 * pi))
        out.append(np.linalg.norm(om))
    return np.array(out, dtype=np.float64)


def component_descriptors(mask, zooms=(1.0, 1.0, 1.0), order=10, max_vox=6000, rng=None):
    idx = np.argwhere(mask)
    if len(idx) > max_vox:
        rng = rng or np.random.default_rng(0); idx = idx[rng.choice(len(idx), max_vox, replace=False)]
    coords = idx.astype(np.float64) * np.asarray(zooms, dtype=np.float64)
    return descriptors(coords, np.full(len(idx), float(np.prod(zooms))), order=order)


def _selftest():
    rng = np.random.default_rng(1)
    g = np.zeros((32, 32, 32), bool); zz, yy, xx = np.mgrid[:32, :32, :32]
    g |= ((zz - 14) ** 2 / 36 + (yy - 16) ** 2 / 16 + (xx - 18) ** 2 / 9) < 1   # an ellipsoid
    g |= ((zz - 22) ** 2 + (yy - 10) ** 2 + (xx - 12) ** 2) < 12                 # plus a small ball
    d0 = component_descriptors(g, max_vox=10 ** 6)
    for k, axes in ((1, (0, 1)), (2, (1, 2)), (3, (0, 2))):
        d1 = component_descriptors(np.rot90(g, k, axes), max_vox=10 ** 6)
        print(f"rot90 x{k} axes {axes}: max |dF| = {np.abs(d1 - d0).max():.2e}  (rel {np.abs(d1 - d0).max() / d0.max():.1e})")
    ball = ((zz - 16) ** 2 + (yy - 16) ** 2 + (xx - 16) ** 2) < 100
    db = component_descriptors(ball, max_vox=10 ** 6); pairs = nl_pairs(10)
    l0 = max(abs(v) for (n, l), v in zip(pairs, db) if l == 0); lx = max(abs(v) for (n, l), v in zip(pairs, db) if l > 0)
    print(f"ball: max F(l=0) {l0:.3e}, max F(l>0) {lx:.3e}  (l>0 should be ~0; {len(pairs)} descriptors)")


if __name__ == "__main__":
    _selftest()
