#!/usr/bin/env python3
"""
pregenerate_validation_set.py
===============================
Draws N fresh (non-cheap-repeat) cases from paired_pathology_generator and
saves each clean+pathological pair to disk, so the SAME 2*N images can be
evaluated at every checkpoint throughout training -- the fixed benchmark
needed to tell epistemic uncertainty (would more training help?) apart from
aleatoric noise (is the per-checkpoint metric just measuring which random
case got drawn?). See the chat discussion this came out of for the full
reasoning; short version: right now every dice number in the training log
comes from one freshly-drawn case per checkpoint, so a checkpoint "getting
worse" is indistinguishable from "drew a harder case this time." Evaluating
the identical set every time removes that confound.

Mirrors crdmw_train.py's own generator initialization exactly (same
kwargs, same LABEL_LIST_SEGMENTATION, same random_kill_labels, etc.) so the
validation set is drawn from the same distribution as training -- just
frozen once N distinct cases are collected, and never redrawn afterward.

USAGE
-----
    python3 pregenerate_validation_set.py \
        --data_dir /path/to/brain_mris_QCed \
        --out_dir ./validation_set --n 1000 --healthy --coupling

Do a small test run first to check actual per-case disk usage before
committing to the full 1000:
    python3 pregenerate_validation_set.py ... --out_dir ./val_test --n 10
    du -sh ./val_test    # extrapolate to n=1000 from this

Each case is saved as val_case_%04d.pt (torch.save of a plain dict) --
readable independently by anything, not just this script's own loader.
"""
import argparse
import os
import time

import numpy as np
import torch

from generator import paired_pathology_generator
from util import make_mode_sampler

# Same as crdmw_train.py -- kept identical so the validation set is drawn
# from the same label space as training.
LABEL_LIST_SEGMENTATION = [0, 11, 12, 13, 16, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                            44, 46, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 17, 47, 49, 51, 53, 55,
                            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 48, 50, 52, 54, 56]


def build_generator(data_dir, device, healthy_only, coupling, k_rose_range=(1.0, 2.5), mode_weights=None, size=None):
    def k_rose_sampler():
        return float(np.random.uniform(*k_rose_range))

    mode_sampler = make_mode_sampler(restricted=False)
    return paired_pathology_generator(
        data_dir,
        mode_sampler,
        label_list_segmentation=LABEL_LIST_SEGMENTATION,
        n_neutral_labels=20,
        label_list_normalization=[1, 18],
        sides='both',
        labels_to_kill=[45],
        list_to_kill_photo=[],
        list_to_kill_photo_inc_ventricles=[],
        flipping=True,
        mix_real_images=False,
        susceptibility=False,
        couple_appearance=coupling,
        susceptibility_prob=1.0,
        complex_resample=False,
        gibbs_ringing_filter=False,
        generate_pathology=True,
        minimal_io=True,
        compact_labels=True,
        random_kill_labels=list(range(34, 47)),
        random_kill_prob=0.5,
        random_kill_frac=(1.0, 1.0),
        device=device,
        **({} if size is None else {"siz": [size, size, size]}),
        healthy_only=healthy_only,
        k_rose_sampler=k_rose_sampler,
        **({} if mode_weights is None else {"mode_weights": mode_weights}),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--size", type=int, default=None, help="output cube side in voxels (default: the generator's 160; 224 holds a whole brain for figures, 2026-10-13)")
    ap.add_argument("--healthy", action="store_true")
    ap.add_argument("--coupling", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sigma_cap", type=float, default=None,
                    help="override generator.LESION_SIGMA_CAP_MULTIPLE for this draw")
    ap.add_argument("--texture_spread", type=float, default=None,
                    help="override generator.LESION_TEXTURE_SPREAD_MAX for this draw")
    ap.add_argument("--sigma_frac", type=float, default=None,
                    help="SET lesion sigma = this x reference sigma (generator.LESION_SIGMA_FRACTION)")
    ap.add_argument("--force_rings", type=int, default=None, choices=[0, 1],
                    help="generator.LESION_FORCE_RINGS: 1 = every focal lesion is a core/shell ring (old live default)")
    ap.add_argument("--max_zones", type=int, default=None,
                    help="generator.LESION_MAX_ZONES: cap GMM zones per lesion (1 = homogeneous)")
    ap.add_argument("--clean_ref_smooth", type=float, default=None,
                    help="generator.LESION_CLEAN_REF_SMOOTH_SIGMA: blur the clean reference inside the lesion")
    ap.add_argument("--extent_scale", type=str, default=None,
                    help="tract,vascular linear scale on the geometric prior, e.g. 0.64,0.70")
    ap.add_argument("--no_diffuse", action="store_true", help="as train.py")
    ap.add_argument("--atrophy_coupled", action="store_true", help="as train.py")
    ap.add_argument("--displace_clean", action="store_true", help="as train.py")
    ap.add_argument("--tissue_sub", type=str, default=None, help="as train.py")
    ap.add_argument("--lesion_comp", type=str, default=None, help="as train.py, e.g. vascular:0.5,local:0.5,tract:0.3")
    ap.add_argument("--local_hemispheric", type=float, default=0.0, help="as train.py")
    ap.add_argument("--lesion_comp_f", type=str, default=None, help="as train.py, LO,HI")
    ap.add_argument("--comp_min_k", type=float, default=None, help="as train.py")
    ap.add_argument("--contrast_t1", action="store_true", help="as train.py: fixed T1 contrast")
    ap.add_argument("--contrast_t2", action="store_true", help="as train.py: fixed T2 contrast (CSF>GM>WM); for the in-domain T2w gate")
    ap.add_argument("--gmm_sigma_rel", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="as train.py: per-class GMM sigma = class mean x U[MIN,MAX]; for the in-domain texture gate")
    ap.add_argument("--gmm_skew", type=float, default=0.0, help="as train.py")
    ap.add_argument("--noise_std", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="as train.py: acquisition-grid noise std U[MIN,MAX] of the WM mean (generator default 0.05 0.15); for the in-domain noise gate")
    ap.add_argument("--t2_prob", type=float, default=0.0, help="as train.py: per-image probability of a T2-realistic contrast")
    ap.add_argument("--seed", type=int, default=None, help="torch/cuda/numpy seed (unseeded by default)")
    ap.add_argument("--mode_polarity", action="store_true", help="as train.py")
    ap.add_argument("--gmm_mean_min", type=float, default=None, help="as train.py")
    ap.add_argument("--dark_csf", type=float, default=0.0, help="as train.py")
    ap.add_argument("--concentric_zones", action="store_true", help="as train.py")
    ap.add_argument("--atrophy_prob", type=float, default=None, help="as train.py")
    ap.add_argument("--adni", action="store_true", help="as train.py")
    ap.add_argument("--visibility_debug", action="store_true",
                    help="print, from inside the visibility correction, the globals it sees and the branch taken")
    ap.add_argument("--max_draws", type=int, default=None,
                     help="Safety cap on total draws (fresh + cheap-repeat skipped). "
                          "Defaults to 5*n -- fresh-draw fraction is usually well above 20%%, "
                          "so this should rarely bind; raise it if it does.")
    args = ap.parse_args()
    import generator as _gen
    if args.sigma_cap is not None: _gen.LESION_SIGMA_CAP_MULTIPLE = args.sigma_cap
    if args.texture_spread is not None: _gen.LESION_TEXTURE_SPREAD_MAX = args.texture_spread
    if args.sigma_frac is not None: _gen.LESION_SIGMA_FRACTION = args.sigma_frac
    if args.force_rings is not None: _gen.LESION_FORCE_RINGS = bool(args.force_rings)
    if args.max_zones is not None: _gen.LESION_MAX_ZONES = args.max_zones
    if args.clean_ref_smooth is not None: _gen.LESION_CLEAN_REF_SMOOTH_SIGMA = args.clean_ref_smooth
    if args.visibility_debug: _gen.VISIBILITY_DEBUG = True
    if args.displace_clean: _gen.DISPLACE_CLEAN = True
    if args.adni: _gen.HEALTHY_EXTRA_DATASETS = ('ADNI3', 'OASIS3')
    if args.concentric_zones: _gen.CONCENTRIC_ZONES = True
    if args.gmm_mean_min is not None: _gen.GMM_MEAN_MIN = args.gmm_mean_min
    if args.dark_csf > 0: _gen.DARK_CSF_PROB = args.dark_csf
    if args.mode_polarity: _gen.POLARITY_BY_MODE = {"vascular": {"t1": 0.10, "t2": 0.85}, "local": {"t1": 0.25, "t2": 0.85}, "tract": {"t1": 0.15, "t2": 0.95}}
    if args.tissue_sub: _gen.TISSUE_SUB = {k: float(v) for k, v in (kv.split(":") for kv in args.tissue_sub.split(","))}
    if args.lesion_comp: _gen.LESION_COMPOSITION = {k: float(v) for k, v in (kv.split(":") for kv in args.lesion_comp.split(","))}
    if args.lesion_comp_f: _gen.COMP_F_CSF = tuple(float(v) for v in args.lesion_comp_f.split(","))
    if args.comp_min_k is not None: _gen.COMP_MIN_K = float(args.comp_min_k)
    if getattr(args,"contrast_t1",False): _gen.CONTRAST_T1 = True
    if getattr(args,"contrast_t2",False): _gen.CONTRAST_T2 = True
    if getattr(args,"t2_prob",0.0) > 0: _gen.T2_CONTRAST_PROB = args.t2_prob
    if args.local_hemispheric > 0:
        import pathology_primitives as _pp
        _pp.LOCAL_HEMISPHERIC_PROB = args.local_hemispheric
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
    mode_weights = None
    if args.no_diffuse or args.atrophy_coupled:
        import pathology_primitives as _pp
        _pp.DISABLED_MODES = {"Global Pathology"}
        if not args.atrophy_coupled:
            _pp.DISABLED_MODES.add("Geometric Atrophy")
        else:
            _pp.ATROPHY_VENTRICLE_PROB = 0.5
        if args.atrophy_prob is not None: _pp.COUPLE_PROB_BY_MODE["Geometric Atrophy"] = args.atrophy_prob
        mode_weights = (0.23, 0.0, 0.0, 0.385, 0.385)
    if args.extent_scale is not None:
        import pathology_primitives as _pp
        _t, _v = (float(x) for x in args.extent_scale.split(","))
        _pp.LESION_EXTENT_SCALE = {"tract": _t, "vascular": _v}
        print(f"extent scale: tract={_t} vascular={_v}")
    if args.gmm_sigma_rel is not None:
        _lo, _hi = float(args.gmm_sigma_rel[0]), float(args.gmm_sigma_rel[1])
        assert 0.0 < _lo <= _hi, args.gmm_sigma_rel
        assert hasattr(_gen, "GMM_SIGMA_REL"), "generator has no GMM_SIGMA_REL (stale snapshot?)"
        _gen.GMM_SIGMA_REL = (_lo, _hi)
        print(f"[generator] per-class GMM sigma = mean x U[{_lo}, {_hi}] (default absolute U[5,25])")
    if args.gmm_skew > 0:
        assert 0.0 < args.gmm_skew < 1.0, args.gmm_skew
        assert hasattr(_gen, "GMM_SKEW"), "generator has no GMM_SKEW (stale snapshot?)"
        _gen.GMM_SKEW = float(args.gmm_skew)
        print(f"[generator] gmm_skew a={args.gmm_skew}")
    if args.noise_std is not None:
        _lo, _hi = float(args.noise_std[0]), float(args.noise_std[1])
        assert 0.0 <= _lo <= _hi, args.noise_std
        assert hasattr(_gen, "NOISE_STD_MIN"), "generator has no NOISE_STD_MIN (stale snapshot?)"
        _gen.NOISE_STD_MIN, _gen.NOISE_STD_MAX = _lo, _hi
        print(f"[generator] acquisition noise std U[{_lo}, {_hi}] of WM (default U[0.05, 0.15])")
    print(f"lesion texture: sigma_cap={_gen.LESION_SIGMA_CAP_MULTIPLE} sigma_frac={_gen.LESION_SIGMA_FRACTION} texture_spread_max={_gen.LESION_TEXTURE_SPREAD_MAX} force_rings={_gen.LESION_FORCE_RINGS} max_zones={_gen.LESION_MAX_ZONES} clean_ref_smooth={_gen.LESION_CLEAN_REF_SMOOTH_SIGMA}")
    print(f"composition={_gen.LESION_COMPOSITION} local_hemispheric={args.local_hemispheric} seed={args.seed}")

    os.makedirs(args.out_dir, exist_ok=True)
    max_draws = args.max_draws or (5 * args.n)

    existing = sorted(f for f in os.listdir(args.out_dir)
                       if f.startswith("val_case_") and f.endswith(".pt"))
    start_idx = len(existing)
    if start_idx >= args.n:
        print(f"Already have {start_idx}/{args.n} cases in {args.out_dir} -- nothing to do.")
        return
    if start_idx > 0:
        print(f"Found {start_idx} existing cases -- generating {args.n - start_idx} more "
              f"(assumes existing files are the first {start_idx}, numbered contiguously from 0, "
              f"which is how this script always writes them).")

    gen = build_generator(args.data_dir, args.device, args.healthy, args.coupling, mode_weights=mode_weights, size=args.size)

    t0 = time.time()
    saved = start_idx
    drawn = 0
    skipped_repeat = 0
    while saved < args.n and drawn < max_draws:
        case = next(gen)
        drawn += 1
        if case.get('is_cheap_repeat', False):
            skipped_repeat += 1
            continue

        _pm = case.get('pathology_metadata', {})
        record = {
            'image_clean': case['image_clean'].detach().cpu(),
            'image_path': case['image_path'].detach().cpu(),
            'seg_clean_idx': case['seg_clean_idx'].detach().cpu(),
            'seg_path_idx': case['seg_path_idx'].detach().cpu(),
            'k_rose': case.get('k_rose'),
            'mode_name': _pm.get('mode_name', 'unknown'),
            'tissue_sub': _pm.get('tissue_sub', None),
            'composition': _pm.get('composition', None),
            'composition_drawn': _pm.get('composition_drawn', None),
            'composition_skipped': _pm.get('composition_skipped', None),
            'visibility_events': _pm.get('visibility_events', None),
            'suppressed_share': _pm.get('suppressed_share', None),
            'sd_surround': _pm.get('sd_surround', None),
            'tau': _pm.get('tau', None),
        }
        out_path = os.path.join(args.out_dir, f"val_case_{saved:04d}.pt")
        torch.save(record, out_path)
        saved += 1

        newly_saved = saved - start_idx
        if newly_saved % 50 == 0 or saved == args.n:
            elapsed = time.time() - t0
            print(f"[{saved}/{args.n}] saved  ({drawn} total draws this run, {skipped_repeat} "
                  f"cheap-repeats skipped, {elapsed:.0f}s elapsed, "
                  f"{elapsed/max(newly_saved,1):.2f}s/case)")

    if saved < args.n:
        print(f"WARNING: only saved {saved}/{args.n} cases after {drawn} draws "
              f"(hit --max_draws={max_draws}). Raise --max_draws or check the "
              f"cheap-repeat rate.")

    total_bytes = sum(os.path.getsize(os.path.join(args.out_dir, f))
                       for f in os.listdir(args.out_dir) if f.endswith('.pt'))
    print(f"\nDone. {saved} cases saved to {args.out_dir}")
    print(f"Total size: {total_bytes / 1e9:.2f} GB  "
          f"({total_bytes / max(saved, 1) / 1e6:.1f} MB/case average)")
    print(f"Wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()