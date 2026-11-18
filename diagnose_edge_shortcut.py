#!/usr/bin/env python3
"""
diagnose_edge_shortcut.py
==========================
Tests, quantitatively, whether the regression head has learned a generic
"high local intensity gradient" shortcut instead of genuine pathology
localization -- the hypothesis raised by seeing it fire on the skull,
brain edge, and deep gray matter boundaries, all high-contrast anatomical
edges, on real T1 data.

The core test: on the CLEAN stream specifically, the true regression
target is EXACTLY zero everywhere (no pathology exists there at all, by
construction). If |regression output| correlates strongly with local
image gradient magnitude on this stream, that is direct, quantitative
evidence the head is keying off generic edges rather than pathology --
there is no pathology for it to be legitimately detecting, so any
gradient-shaped output there is definitionally spurious.

As a second, complementary check: on the PATHOLOGICAL stream, compares
how well |regression output| tracks the TRUE regression target vs. how
well it tracks local gradient magnitude. If it tracks gradient magnitude
similarly well or better than it tracks the true target, that's evidence
the "signal" visible in the pathological stream is also largely
edge-driven rather than pathology-specific, not a separate, healthier
mechanism from what's happening on the clean stream.

USAGE
-----
    python3 diagnose_edge_shortcut.py --checkpoint path/to/synthseg_XXXXX.pth \\
        --validation-dir path/to/validation_set --n-cases 50

Requires networks.py to be importable (same directory or on PYTHONPATH).
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

from networks import load_checkpoint


def gradient_magnitude_3d(x):
    """Simple finite-difference gradient magnitude for a (1,1,D,H,W) volume.

    Not a Sobel filter -- central differences are enough to test for
    correlation with a smooth, blurred regression output; a sharper edge
    operator would only matter if testing against raw, unblurred pixel
    noise, which isn't the comparison here.
    """
    gx = x[:, :, 2:, :, :] - x[:, :, :-2, :, :]
    gy = x[:, :, :, 2:, :] - x[:, :, :, :-2, :]
    gz = x[:, :, :, :, 2:] - x[:, :, :, :, :-2]
    # Crop all three to the shared valid region (D-2, H-2, W-2) so they add cleanly.
    gx = gx[:, :, :, 1:-1, 1:-1]
    gy = gy[:, :, 1:-1, :, 1:-1]
    gz = gz[:, :, 1:-1, 1:-1, :]
    return torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-12)


def pearson_corr(a, b):
    """Pearson correlation between two flattened tensors."""
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-8)
    return (a @ b / denom).item()


def partial_corr(r_xy, r_xz, r_yz):
    """Partial correlation of X and Y controlling for Z, from the three
    pairwise correlations, via the standard formula. Answers: how much of
    X and Y's correlation survives once Z's own relationship to both is
    removed -- e.g. does reg_out (X) still track the true target (Y) once
    gradient magnitude (Z) is accounted for, or does controlling for
    gradient explain the whole thing away?
    """
    denom = np.sqrt(max((1 - r_xz**2) * (1 - r_yz**2), 1e-12))
    return (r_xy - r_xz * r_yz) / denom


def normalize_per_sample(x):
    """Matches train.py's normalize_per_sample exactly -- see that file for
    the full derivation. Reimplemented here (not imported) so this script
    has no dependency on train.py's argparse-heavy module-level code
    running at import time."""
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        xi = x[i]
        fg = xi[xi > 0]
        if fg.numel() > 0:
            p_lo, p_hi = torch.quantile(fg, torch.tensor([0.01, 0.99], device=x.device, dtype=fg.dtype))
        else:
            p_lo, p_hi = torch.tensor(0.0, device=x.device), torch.tensor(1.0, device=x.device)
        xi = xi.clamp(min=p_lo, max=p_hi) - p_lo
        scale = (p_hi - p_lo).clamp(min=1e-7)
        out[i] = (xi / scale).clamp(0, 1)
    return out


def run_diagnostic(checkpoint_path, validation_dir, device="cuda", n_cases=50):
    model, _, _ = load_checkpoint(checkpoint_path, device, verbose=True)
    model.eval()

    case_paths = sorted(
        os.path.join(validation_dir, f) for f in os.listdir(validation_dir)
        if f.startswith("val_case_") and f.endswith(".pt")
    )[:n_cases]
    if not case_paths:
        raise ValueError(f"No val_case_*.pt files found in {validation_dir}")
    print(f"Running on {len(case_paths)} validation cases\n")

    clean_edge_corrs = []
    clean_intensity_corrs = []
    path_target_corrs = []
    path_edge_corrs = []
    path_target_edge_corrs = []   # true_target vs gradient -- needed to interpret path_target_corrs properly
    path_partial_corrs = []       # reg_out vs true_target, CONTROLLING FOR gradient

    with torch.no_grad():
        for p in case_paths:
            r = torch.load(p, map_location="cpu")
            img_clean = r["image_clean"].to(device).float().unsqueeze(0).unsqueeze(0)
            img_path = r["image_path"].to(device).float().unsqueeze(0).unsqueeze(0)

            clean_norm = normalize_per_sample(img_clean)
            path_norm = normalize_per_sample(img_path)

            _, reg_clean = model(clean_norm)
            _, reg_path = model(path_norm)

            # --- Test 1: clean stream, |reg_out| vs local gradient magnitude ---
            grad_clean = gradient_magnitude_3d(img_clean)
            reg_clean_cropped = reg_clean[:, :, 1:-1, 1:-1, 1:-1]
            clean_edge_corrs.append(
                pearson_corr(reg_clean_cropped.abs(), grad_clean))

            # --- Test 1b (thresholding shortcut): clean stream, |reg_out| vs
            # distance from mid-range normalized intensity. Same logic as
            # Test 1 -- true target is exactly zero everywhere here, so any
            # correlation with "how extreme is this intensity" (as opposed
            # to "how sharp is the local edge") is also definitionally
            # spurious, and tests a DIFFERENT candidate shortcut: simple
            # intensity thresholding rather than edge/gradient detection.
            clean_norm_cropped = clean_norm[:, :, 1:-1, 1:-1, 1:-1]
            clean_intensity_corrs.append(
                pearson_corr(reg_clean_cropped.abs(), (clean_norm_cropped - 0.5).abs()))

            # --- Pathological stream ---
            fg = img_path.clamp(min=0)
            fg = fg[fg > 0]
            if fg.numel() > 0:
                p_lo, p_hi = torch.quantile(fg, torch.tensor([0.01, 0.99], device=device))
            else:
                p_lo, p_hi = torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
            scale = (p_hi - p_lo).clamp(min=1e-6)
            true_reg_target = torch.tanh((img_path - img_clean) / scale)

            grad_path = gradient_magnitude_3d(img_path)
            reg_path_cropped = reg_path[:, :, 1:-1, 1:-1, 1:-1]
            true_target_cropped = true_reg_target[:, :, 1:-1, 1:-1, 1:-1]

            r_xy = pearson_corr(reg_path_cropped.abs(), true_target_cropped.abs())
            r_xz = pearson_corr(reg_path_cropped.abs(), grad_path)
            # This is the piece the original version of this script never
            # computed, and without it Test 2 alone is not interpretable:
            # if the true target is ITSELF edge-correlated (the halo effect
            # from blur/resample spreading a lesion's difference into a
            # boundary-shaped region), then a pure edge-detector with ZERO
            # genuine pathology signal would still show a substantial raw
            # correlation with the true target, for free, just by riding
            # on that shared edge structure -- verified directly on
            # synthetic data where the answer is known: a pure
            # edge-detector produced raw corr(reg_out, true_target)=0.441
            # (which would look like real success on its own) that
            # collapsed to a partial correlation of -0.002 once gradient
            # was controlled for.
            r_yz = pearson_corr(true_target_cropped.abs(), grad_path)

            path_target_corrs.append(r_xy)
            path_edge_corrs.append(r_xz)
            path_target_edge_corrs.append(r_yz)
            path_partial_corrs.append(partial_corr(r_xy, r_xz, r_yz))

    clean_edge_corrs = np.array(clean_edge_corrs)
    clean_intensity_corrs = np.array(clean_intensity_corrs)
    path_target_corrs = np.array(path_target_corrs)
    path_edge_corrs = np.array(path_edge_corrs)
    path_target_edge_corrs = np.array(path_target_edge_corrs)
    path_partial_corrs = np.array(path_partial_corrs)

    print("=" * 70)
    print("TEST 1: clean stream -- |reg_out| vs local gradient magnitude")
    print("        (true target is EXACTLY zero everywhere here)")
    print(f"  mean r = {clean_edge_corrs.mean():+.3f}  (r^2 = {clean_edge_corrs.mean()**2:.3f}, i.e. "
          f"{clean_edge_corrs.mean()**2*100:.1f}% of variance explained)   n={len(clean_edge_corrs)}")
    print()
    print("TEST 1b: clean stream -- |reg_out| vs |normalized intensity - 0.5|")
    print("         (the thresholding-shortcut candidate, as distinct from edges)")
    print(f"  mean r = {clean_intensity_corrs.mean():+.3f}  (r^2 = {clean_intensity_corrs.mean()**2:.3f}, "
          f"{clean_intensity_corrs.mean()**2*100:.1f}% of variance explained)")
    print()
    print("TEST 2: pathological stream -- |reg_out| vs TRUE regression target (raw)")
    print(f"  mean r = {path_target_corrs.mean():+.3f}")
    print()
    print("TEST 3: pathological stream -- |reg_out| vs local gradient magnitude")
    print(f"  mean r = {path_edge_corrs.mean():+.3f}")
    print()
    print("TEST 4 (new): pathological stream -- TRUE target vs local gradient magnitude")
    print("              (how edge-like is the true target ITSELF, e.g. from the halo effect)")
    print(f"  mean r = {path_target_edge_corrs.mean():+.3f}")
    print()
    print("TEST 5 (new, the important one): PARTIAL correlation of reg_out with TRUE")
    print("        target, CONTROLLING FOR gradient -- how much of Test 2's raw 0.454-style")
    print("        correlation survives once shared edge structure is accounted for")
    print(f"  mean partial r = {path_partial_corrs.mean():+.3f}   median = {np.median(path_partial_corrs):+.3f}")
    print("=" * 70)
    print()
    print(f"For reference, Test 2's raw correlation was {path_target_corrs.mean():+.3f}. If the partial")
    print(f"correlation above ({path_partial_corrs.mean():+.3f}) is close to that raw value, the model has")
    print("real, pathology-specific signal beyond generic edge-riding. If it collapsed toward")
    print("zero, Test 2's apparent success is substantially or entirely explained by the true")
    print("target's own edge-like structure combined with the model doing little more than")
    print("edge detection -- verified on synthetic data to produce exactly this pattern.")

    return dict(clean_edge_corrs=clean_edge_corrs, clean_intensity_corrs=clean_intensity_corrs,
                path_target_corrs=path_target_corrs, path_edge_corrs=path_edge_corrs,
                path_target_edge_corrs=path_target_edge_corrs, path_partial_corrs=path_partial_corrs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--validation-dir", required=True)
    ap.add_argument("--n-cases", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_diagnostic(args.checkpoint, args.validation_dir, device=args.device, n_cases=args.n_cases)