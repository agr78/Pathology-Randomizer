"""Per-image, disease-agnostic thresholds for a signed anomaly map.

The scan's own healthy tissue is the null: its centre and scale are estimated
robustly (median, MAD) from the signed map inside the brain, so the head's
constant offset on healthy tissue (+0.14 on some arms) is removed rather than
thresholded. The free constant is a statistical one -- an FDR level q -- not a
z, a prevalence, or anything about the disease.

rule "z"   : |dev| > z * sigma                       (Hampel; z=3 is the convention)
rule "fdr" : Benjamini-Hochberg at level q on two-sided p-values under the
             fitted null. The BH cut RISES with the number of small p-values,
             so the threshold adapts to how much signal the scan contains
             without knowing the prevalence. Voxels are positively dependent
             (PRDS), under which BH stays valid, if conservative.
null "gaussian": p = 2(1 - Phi(|dev|/sigma)), sigma = 1.4826 MAD
null "laplace" : p = exp(-|dev|/b),           b = MAD / ln 2   (heavier tail)
"""
import numpy as np
from scipy.stats import norm

LN2 = np.log(2.0)


def fit_null(x):
    mu = float(np.median(x)); mad = float(np.median(np.abs(x - mu))) + 1e-9
    return mu, mad


def pvalues(dev, mad, null):
    a = np.abs(dev)
    if null == "gaussian":
        return 2.0 * norm.sf(a / (1.4826 * mad))
    if null == "laplace":
        return np.exp(-a / (mad / LN2))
    if null == "twopiece":
        # each side gets its own scale from its own deviations (median |dev| of
        # that side); a skewed healthy distribution is then not read as signal on
        # its heavy side. Still polarity-free: both sides are treated identically.
        sr = 1.4826 * (np.median(dev[dev > 0]) + 1e-9); sl = 1.4826 * (np.median(-dev[dev < 0]) + 1e-9)
        return 2.0 * norm.sf(np.where(dev > 0, a / sr, a / sl))
    raise ValueError(null)


def bh_cut(p, q):
    """Largest p-value passing Benjamini-Hochberg at level q, or None."""
    ps = np.sort(p); m = ps.size
    ok = np.nonzero(ps <= (np.arange(1, m + 1) / m) * q)[0]
    return float(ps[ok[-1]]) if ok.size else None


def threshold(sreg_brain, rule="z", z=3.0, q=0.05, null="gaussian"):
    """Return (mu, cut) with positives = |sreg - mu| >= cut; cut=inf if nothing passes."""
    mu, mad = fit_null(sreg_brain)
    if rule == "z":
        return mu, z * 1.4826 * mad
    dev = sreg_brain - mu
    p = pvalues(dev, mad, null)
    pc = bh_cut(p, q)
    if pc is None:
        return mu, np.inf
    return mu, float(np.abs(dev)[p <= pc].min())      # NB: for twopiece the cut differs per side; see threshold_sides


def threshold_sides(sreg_brain, q=0.05, null="twopiece"):
    """(mu, cut_bright, cut_dark) with positives dev >= cut_bright or dev <= -cut_dark."""
    mu, mad = fit_null(sreg_brain); dev = sreg_brain - mu
    p = pvalues(dev, mad, null); pc = bh_cut(p, q)
    if pc is None:
        return mu, np.inf, np.inf
    ok = p <= pc
    cb = float(dev[ok & (dev > 0)].min()) if (ok & (dev > 0)).any() else np.inf
    cd = float(-dev[ok & (dev < 0)].max()) if (ok & (dev < 0)).any() else np.inf
    return mu, cb, cd


def cluster_mask(dev, brain, sigma, zc=2.0, min_vox=27):
    """Blind spatial rule: cluster-forming cut |dev| >= zc*sigma inside the brain,
    keep 26-connected components with >= min_vox voxels (a physical minimum
    lesion size, not a statistical level). Returns (bright, dark, abs) masks."""
    from scipy import ndimage
    out = {}
    for name, m in (("bright", dev >= zc * sigma), ("dark", dev <= -zc * sigma), ("abs", np.abs(dev) >= zc * sigma)):
        m = m & brain
        lab, n = ndimage.label(m, structure=np.ones((3, 3, 3)))
        if n:
            sizes = np.bincount(lab.ravel()); keep = sizes >= min_vox; keep[0] = False
            m = keep[lab]
        out[name] = m
    return out["bright"], out["dark"], out["abs"]
