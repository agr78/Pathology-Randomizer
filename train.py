import os
import glob
import time
import argparse
import math
import subprocess
import warnings
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")   # headless: a Tk backend under DISPLAY crashed a local run (Tcl_AsyncDelete, 2026-09-30)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from generator import paired_pathology_generator
import pathology_primitives
from util import make_mode_sampler, get_slices
from networks import SynthSegUNet
from prefetch import Prefetcher
from kspace_ops import simulate_kspace_motion
import live_diagnostics as diag
from target_utils import apply_target_region, load_weights_keep_head, freeze_trunk, frozen_bn_eval, frozen_bn_batch, is_reg_head_key


def _git_identifier():
    # Output directory naming now comes from the commit this script is
    # actually running from, not a hand-picked, ever-accumulating prefix
    # (mcp -> wce -> mdmw -> crdmw was a workaround for not having this).
    # A worktree checked out to a specific commit gets a permanently
    # distinct, unambiguous output location automatically -- no manual
    # rename needed for the next run. Falls back to "nogit" rather than
    # crashing if this ever runs outside a git checkout.
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=script_dir, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BASE_COMMIT")   # snapshots outside a checkout
    try:
        return open(base).read().split()[0][:7]
    except Exception:
        return "nogit"

# --- FreeSurfer colormap, for visualization only ---
# From brain_mris_QCed.readme.txt, brainseg_with_extracerebral scheme
# (labels 0-56): ids 47/49/51/53/55 (left) and 48/50/52/54/56 (right) are
# HypoThal-noMB/Fornix/MammillaryBody/Basal-Forebrain/SeptalNuc, and ids
# 34-46 are extracerebral tissue (air, artery, eyeball, muscle, etc.).
# Not the standard FreeSurferColorLUT numbering (e.g. standard id 2 =
# Left-Cerebral-White-Matter, this project's id 1 = the same structure) --
# colors are matched by name against the official table, not by id.
FREESURFER_LABELS = {
    0: "Unknown",
    1: "Left-Cerebral-White-Matter", 2: "Left-Cerebral-Cortex", 3: "Left-Lateral-Ventricle",
    4: "Left-Inf-Lat-Vent", 5: "Left-Cerebellum-White-Matter", 6: "Left-Cerebellum-Cortex",
    7: "Left-Thalamus", 8: "Left-Caudate", 9: "Left-Putamen", 10: "Left-Pallidum",
    11: "3rd-Ventricle", 12: "4th-Ventricle", 13: "Brain-Stem", 14: "Left-Hippocampus",
    15: "Left-Amygdala", 16: "CSF", 17: "Left-Accumbens-area", 18: "Right-Cerebral-White-Matter",
    19: "Right-Cerebral-Cortex", 20: "Right-Lateral-Ventricle", 21: "Right-Inf-Lat-Vent",
    22: "Right-Cerebellum-White-Matter", 23: "Right-Cerebellum-Cortex", 24: "Right-Thalamus",
    25: "Right-Caudate", 26: "Right-Putamen", 27: "Right-Pallidum", 28: "Right-Hippocampus",
    29: "Right-Amygdala", 30: "Right-Accumbens-area", 31: "WM-hypointensities", 32: "Optic-Chiasm",
    33: "Lesion",
    34: "air-internal", 35: "artery", 36: "eyeball", 37: "other-tissues", 38: "rectus-muscles",
    39: "mucosa", 40: "skin", 41: "spinal-cord", 42: "vein", 43: "bone-cortical",
    44: "cone-cancellous", 45: "background", 46: "optic-nerve",
    47: "Left-HypoThal-noMB", 48: "Right-HypoThal-noMB", 49: "Left-Fornix", 50: "Right-Fornix",
    51: "Left-MammillaryBody", 52: "Right-MammillaryBody", 53: "Left-Basal-Forebrain",
    54: "Right-Basal-Forebrain", 55: "Left-SeptalNuc", 56: "Right-SeptalNuc",
    57: "Left-choroid-plexus", 58: "Right-choroid-plexus",   # appended 2026-09-29 (--choroid_target), from generation label 9
}

# Verified against the official FreeSurferColorLUT.txt for ids 0-33
# (freesurfer/freesurfer repo, distribution/FreeSurferColorLUT.txt); ids
# 34-56 (extracerebral tissue and the extended subcortical structures) are
# from the user-provided brain_mris_QCed.readme.txt directly. RGB as
# published/provided (0-255).
FREESURFER_LUT_RGB = {
    "Unknown": (0, 0, 0),
    "Left-Cerebral-White-Matter": (245, 245, 245), "Left-Cerebral-Cortex": (205, 62, 78),
    "Left-Lateral-Ventricle": (120, 18, 134), "Left-Inf-Lat-Vent": (196, 58, 250),
    "Left-Cerebellum-White-Matter": (220, 248, 164), "Left-Cerebellum-Cortex": (230, 148, 34),
    "Left-Thalamus": (0, 118, 14), "Left-Caudate": (122, 186, 220), "Left-Putamen": (236, 13, 176),
    "Left-Pallidum": (12, 48, 255), "3rd-Ventricle": (204, 182, 142), "4th-Ventricle": (42, 204, 164),
    "Brain-Stem": (119, 159, 176), "Left-Hippocampus": (220, 216, 20), "Left-Amygdala": (103, 255, 255),
    "CSF": (60, 60, 60), "Left-Accumbens-area": (255, 165, 0),
    "Right-Cerebral-White-Matter": (245, 245, 245), "Right-Cerebral-Cortex": (205, 62, 78),
    "Right-Lateral-Ventricle": (120, 18, 134), "Right-Inf-Lat-Vent": (196, 58, 250),
    "Right-Cerebellum-White-Matter": (220, 248, 164), "Right-Cerebellum-Cortex": (230, 148, 34),
    "Right-Thalamus": (0, 118, 14), "Right-Caudate": (122, 186, 220), "Right-Putamen": (236, 13, 176),
    "Right-Pallidum": (13, 48, 255), "Right-Hippocampus": (220, 216, 20), "Right-Amygdala": (103, 255, 255),
    "Right-Accumbens-area": (255, 165, 0), "WM-hypointensities": (200, 70, 255), "Optic-Chiasm": (234, 169, 30),
    "Lesion": (255, 165, 0),
    "air-internal": (0, 170, 0), "artery": (204, 0, 0), "eyeball": (230, 189, 66),
    "other-tissues": (250, 250, 51), "rectus-muscles": (190, 50, 73), "mucosa": (255, 160, 188),
    "skin": (190, 156, 120), "spinal-cord": (0, 187, 169),
    "vein": (0, 84, 255),  # source text showed '0 84 2 55 0' (5 numbers, not the usual 4) --
                            # reconstructed as (0,84,255) assuming a split '255'; worth confirming
                            # against your original file
    "bone-cortical": (238, 215, 150),
    "cone-cancellous": (237, 199, 154),  # 'cone' reads like a possible typo for 'bone-cancellous'
                                          # (cortical/cancellous is the standard bone pairing) but used
                                          # exactly as given rather than silently 'corrected'
    "background": (0, 168, 255), "optic-nerve": (0, 191, 122),
    "Left-HypoThal-noMB": (0, 80, 0), "Right-HypoThal-noMB": (15, 165, 15),
    "Left-Fornix": (0, 255, 255), "Right-Fornix": (27, 187, 253), "Left-MammillaryBody": (178, 255, 80),
    "Right-MammillaryBody": (25, 246, 80), "Left-Basal-Forebrain": (170, 85, 0), "Right-Basal-Forebrain": (139, 69, 19),
    "Left-SeptalNuc": (255, 170, 255), "Right-SeptalNuc": (191, 127, 191),
    "Left-choroid-plexus": (0, 200, 200), "Right-choroid-plexus": (0, 200, 221),   # FreeSurferColorLUT 31 / 63
}


def build_fs_colormap(label_list_segmentation, freesurfer_labels=FREESURFER_LABELS, lut_rgb=FREESURFER_LUT_RGB):
    """Builds a matplotlib ListedColormap indexed by COMPACT index (position
    in label_list_segmentation, 0..n_labels-1) -- the same indexing the
    generator's one-hot targets and the model's own predictions already use,
    so it drops directly into imshow(..., vmin=0, vmax=n_labels-1) with no
    extra remapping. Ids without a verified name/color get a muted,
    deliberately-not-vivid placeholder so they read as 'unverified' rather
    than as a confident (and possibly wrong) real answer."""
    colors, unmatched, k = [], [], 0
    for real_id in label_list_segmentation:
        name = freesurfer_labels.get(real_id)
        rgb = lut_rgb.get(name) if name else None
        if rgb is None:
            unmatched.append(real_id)
            h = (k * 47) % 256
            rgb = (120 + h % 60, 120 + (h * 2) % 60, 120 + (h * 3) % 60)
            k += 1
        colors.append(tuple(c / 255.0 for c in rgb))
    if unmatched:
        print(f"[FS colormap] {len(unmatched)} real ids have no verified FreeSurfer color, shown in muted "
              f"placeholder tones: {unmatched}. Fill in FREESURFER_LABELS above from your actual "
              f"brain_mris_QCed.readme.txt for their real colors.")
    return ListedColormap(colors), len(colors)


# --- 0. Setup Argparse for Command Line execution ---
parser = argparse.ArgumentParser(description="Train SynthSeg with optional motion corruption and pathology augmentation.")
parser.add_argument("--do_motion", action="store_true", help="Pass this flag to enable k-space motion corruption.")
parser.add_argument("--motion_prob", type=float, default=0.5, help="Probability (0.0 to 1.0) of applying motion corruption if --do_motion is set.")
parser.add_argument("--do_pathology", action="store_true", help="Pass this flag to train on the pathology-augmented case each step (it's always generated -- this controls whether it's used in the loss).")
parser.add_argument("--pathology_prob", type=float, default=0.5, help="Probability (0.0 to 1.0) of including the pathological case in the loss each step if --do_pathology is set.")
parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in the most recent run directory for this mode (motion/pathology/baseline).")
parser.add_argument("--no_rng_restore", action="store_true",
                    help="With --resume: ignore the RNG state saved in the checkpoint and keep the "
                         "--seed stream (a fork meant to diverge from its ancestor).")
parser.add_argument("--motion_consistency", action="store_true",
                    help="On steps where motion corruption fires, also feed the motion-free "
                         "volume and penalise the difference between the two bottleneck "
                         "encodings. Asks the encoder to be invariant to motion rather than "
                         "merely tolerate it.")
parser.add_argument("--motion_consistency_weight", type=float, default=1.0,
                    help="Weight on the consistency term. Untuned -- 1.0 puts it on the same "
                         "footing as CE+Dice, which is a starting point rather than a "
                         "calibrated value.")
parser.add_argument("--lesion_weight", type=float, default=1.0,
                    help="UNUSED as of the regression-map refactor: the classification target is "
                         "now always seg_clean_idx, which never contains a Lesion class to "
                         "weight. Left defined (rather than removed) so an existing sbatch "
                         "script that still passes it doesn't fail to parse -- the value is "
                         "accepted and silently has no effect.")
parser.add_argument("--seg_weight", type=float, default=1.0,
                    help="Multiplier on the segmentation loss (CE + Dice). 0 = regression-only training.")
parser.add_argument("--seg_clean_only", action="store_true",
                    help="Segmentation loss on the clean row only; the pathological row trains the regression head alone "
                         "(removes the seg head's 'label the lesion as healthy tissue' instruction from the shared trunk).")
parser.add_argument("--seg_mask_lesion", action="store_true",
                    help="Segmentation loss on both rows, but lesion voxels are excluded from CE and Dice on the pathological row.")
parser.add_argument("--regression_weight", type=float, default=1.0,
                    help="Fixed multiplier on the regression loss relative to CE+Dice on the "
                         "healthy segmentation. Used unless --uncertainty_weighting is passed. "
                         "1.0 puts the two on the same footing.")
parser.add_argument("--attention", action="store_true",
                    help="Gate each skip connection on the coarser decoder feature, so wide "
                         "context can suppress or pass fine detail. Changes the parameter set, "
                         "so a run started with it cannot resume from a checkpoint without it.")
parser.add_argument("--attention_dense", action="store_true",
                    help="With --attention: each gate is also conditioned on every COARSER decoder feature, not "
                         "just the adjacent one (att1 gets 4 gating inputs, att2 3, att3 2, att4 1). Each extra "
                         "signal is projected at its own resolution then upsampled and summed into the same "
                         "pre-activation, so no new width appears. Changes which representation controls the gate, "
                         "not the receptive field (measured ~132/140 vox) and not what the head reads. Adds "
                         "att{1,2,3}.wg_extra.* keys, so a run started with it cannot resume from a checkpoint "
                         "without it; identity init still opens every gate at step 0.")
parser.add_argument("--attention_identity_init", action="store_true",
                    help="With --attention: initialise the gates fully open (psi bias 8, weight 0), so a gate-less "
                         "checkpoint forked with --attention starts as the network it was (target_utils.load_weights_keep_head "
                         "loads trunk and head strictly and leaves only the gates fresh).")
parser.add_argument("--task", choices=["segment", "register"], default="segment",
                    help="segment: healthy anatomy plus the regression/probability heads. "
                         "register: two generator draws per step, moving=pathological subject 1 "
                         "and fixed=clean subject 2, supervised by label overlap only.")
parser.add_argument("--smoothness_weight", type=float, default=0.05,
                    help="--task register only: weight on the flow-gradient penalty.")
parser.add_argument("--flow_scale", type=float, default=0.1,
                    help="--task register only: scales the network output into normalized "
                         "grid units, bounding the maximum displacement.")
parser.add_argument("--seed", type=int, default=None,
                    help="Seed torch, numpy and CUDA. Unset leaves every run unseeded, which "
                         "is how all runs before 2026-09-13 were trained.")
parser.add_argument("--seg_prior", action="store_true",
                    help="Feed the segmentation head's softmax into the regression head, so "
                         "intensity is judged against the tissue it sits in. Changes the "
                         "parameter set; detected on load from the head's input width.")
parser.add_argument("--tag", type=str, default=None,
                    help="Appended to the run prefix, so several runs on one commit get their "
                         "own *_out directories instead of sharing one by timestamp.")
parser.add_argument("--max_steps", type=int, default=None,
                    help="Stop cleanly at this step instead of the built-in 500k. For sprints; "
                         "the checkpoint at that step is written before exit.")
parser.add_argument("--no_diffuse", action="store_true",
                    help="Remove Global Pathology and Geometric Atrophy from the pathology draw, "
                         "as primary AND via coupling. They match no evaluation set and were "
                         "the weakest group in-domain.")
parser.add_argument("--atrophy_coupled", action="store_true",
                    help="Global Pathology off; Geometric Atrophy never drawn as the primary mode "
                         "but kept as a coupled secondary on every mode, preferring ventricles "
                         "half the time. Pair with --displace_clean so the displacement is "
                         "anatomy rather than regression target.")
parser.add_argument("--target_lesion_only", action="store_true",
                    help="Zero the regression target outside the lesion label, so the network is "
                         "trained on the lesion rather than the intensity halo around it.")
parser.add_argument("--halo_weight", type=float, default=0.0,
                    help="Extra regression-loss weight on zero-target voxels within HALO_RADIUS of "
                         "the lesion label (the PSF halo). Only meaningful with --target_lesion_only.")
parser.add_argument("--target_region", type=float, default=0.0,
                    help="Inside the lesion label, target = sign x max(|tanh(dI/scale)|, C); sign is the "
                         "voxel's own, or the lesion's mean sign where dI ~ 0. Keeps the geometric label "
                         "(generator.LABEL_SUPPRESS_INVISIBLE = False). Must exceed --target_floor. 0 = off. "
                         "C = 0.18 is the generator's own visibility floor in target units: the Rose threshold at "
                         "the bottom of the k_rose range (k = 1, one surround-SD), tanh(sd_surround / scale), median "
                         "over 199 pregenerated draws with a lesion = 0.184 / 0.182 on two independent sets "
                         "(eval/composition_stage0.py --suppressed_share, 2026-09-22) -- a generator-internal "
                         "constant, no data in it.")
parser.add_argument("--exclude_datasets", nargs="*", default=[],
                    help="Healthy-pool datasets to drop. The base cohorts are assumed healthy at dataset level. "
                         "Buckner40 is no longer in the pool at all (generator.py, 2026-10-03: it spans "
                         "young, old and demented adults); the launchers still pass it here as a second "
                         "line of defence, which is now a harmless no-op.")
parser.add_argument("--adni", action="store_true",
                    help="With --healthy, also admit the cognitively-normal subjects of ADNI3 (Group == CN in its "
                         "participants.tsv). Off by default so baselines keep the young cohorts. ADNI3 only since "
                         "2026-09-30; OASIS3 has its own switch.")
parser.add_argument("--oasis", action="store_true",
                    help="With --healthy, also admit the CDR-0 subjects of OASIS3 (nearest ADRC CDR visit within a year "
                         "of the scan; needs generator.OASIS3_CDR_CSV, else OASIS3 is excluded with a warning).")
parser.add_argument("--oasis_exclude", default=None, metavar="FILE",
                    help="With --oasis: file whose first token per line is a curated OASIS3 id (sub-NNNN) to EXCLUDE from the pool -- "
                         "the held-out healthy cohort (docs/files/oasis3_cohort_triples.txt, 2026-10-07). Without it --oasis admits every "
                         "CDR-0 session and the model cannot be evaluated on that cohort (the generator warns).")
parser.add_argument("--buckner40", action="store_true",
                    help="With --healthy, also admit Buckner40. OFF by default (2026-10-03): the cohort spans "
                         "young, old and DEMENTED adults (Buckner et al. 2004) and ships no diagnosis table, so "
                         "unlike --adni/--oasis it cannot be filtered to healthy subjects and is admitted whole "
                         "or not at all. An explicit --exclude_datasets Buckner40 still overrides this.")
parser.add_argument("--tissue_sub", type=str, default=None,
                    help="Tissue-substitution lesions, per mode: e.g. 'vascular:0.5,local:0.25'. With that "
                         "probability the lesion renders as an existing healthy class (CSF/GM) placed "
                         "where it does not belong; vascular = whole lesion, local = eroded core.")
parser.add_argument("--lesion_comp_f", type=str, default=None,
                    help="generator.COMP_F_CSF: f range of the csf composition zones as LO,HI (default zones 0.7-0.95 / 0.5-0.95); "
                         "f = 1 puts the core at the draw's own CSF level, f > 1 beyond it.")
parser.add_argument("--comp_min_k", type=float, default=None, help="generator.COMP_MIN_K: host-noise visibility gate for composition zones (default 1.0).")
parser.add_argument("--lesion_comp", type=str, default=None,
                    help="generator.LESION_COMPOSITION, per mode: e.g. 'vascular:0.5,local:0.5,tract:0.3'. With that "
                         "probability the lesion is rendered from the draw's own host and CSF/GM class means, "
                         "exempt from the Rose floor and the 2 tau ceiling. Unset = off.")
parser.add_argument("--reg_head", choices=["linear", "mlp"], default="linear",
                    help="Regression head: linear = the 1x1 conv (state-dict regression_conv.*); mlp = "
                         "Conv1x1(in,32) -> ELU -> Conv1x1(32,1) (regression_mlp.*). Detected on load.")
parser.add_argument("--tissue_rel", action="store_true",
                    help="networks.TissueRelative: the regression head also reads (x - median_c) / MAD_c of the "
                         "input under the anatomy head's own class c (detached, per scan). Detected on load.")
parser.add_argument("--no_tissue_rel", action="store_true",
                    help="Ablation (2026-10-06): drop --tissue_rel after parsing, so a launcher whose fixed flag string carries --tissue_rel "
                         "can still fork the own-class-channel ablation. The head then reads the 24 decoder features only and, its shape "
                         "differing from the checkpoint's, starts fresh (pair against the --reinit_head fork).")
parser.add_argument("--reinit_head", action="store_true",
                    help="Discard the checkpoint's regression head on warm start and train a fresh one of the SAME shape (Adam moments restart, as for "
                         "a changed head). The control for every fresh-head arm (tissue_refs / tissue_vec / reg_* change the head's input or shape and "
                         "so re-initialise it): isolates 'fresh head for N steps' from the channel or capacity change. 2026-09-30.")
parser.add_argument("--tissue_refs", type=int, default=0, choices=(0, 3),
                    help="With --tissue_rel: 3 adds per-scan deviation channels against fixed reference tissues "
                         "(white matter, cortex, CSF) for every brain voxel. Detected on load.")
parser.add_argument("--tissue_vec", default="none", choices=("none", "minmax", "full"),
                    help="With --tissue_rel: class-vector channels for the regression head, from every brain voxel scored "
                         "against 11 healthy class pools. minmax = min_c|z_c| and the gap between the two nearest classes "
                         "(2 channels, a segmentation-free intensity prior); full = one signed z_c per class (11). Detected on load.")
parser.add_argument("--tissue_soft", action="store_true",
                    help="Own-class channel as the class-posterior-weighted mixture of per-class deviations instead of the argmax class "
                         "alone (TissueRelative soft=True; needs --tissue_rel and the default pool; 2026-10-07).")
parser.add_argument("--tissue_pool", default="own", choices=("own", "lr", "mirror", "coarse", "coarse_th"),
                    help="With --tissue_rel: the pool each voxel's own-class channel is scored against. own = its 56-class pool; "
                         "lr = class + contralateral homologue; mirror = the contralateral homologue only; coarse = WM/GM/CSF; "
                         "coarse_th = WM/GM/CSF/thalamus. Same channel count (the head warm-starts). Detected on load.")
parser.add_argument("--tissue_unroll", type=int, default=0,
                    help="With --tissue_rel: N re-estimation passes. After each pass the brain voxels whose |r| is a robust outlier "
                         "(median + 3 x 1.4826 MAD of |r|) are dropped from every pool and the channel and head are recomputed; one "
                         "shared head, loss on the final pass. Same channel count (the head warm-starts). Detected on load.")
parser.add_argument("--reg_hidden", type=int, default=32, help="MLP regression head: hidden width (default 32).")
parser.add_argument("--reg_depth", type=int, default=2, help="MLP regression head: number of conv layers (default 2 = conv-ELU-conv).")
parser.add_argument("--reg_kernel", type=int, default=1, choices=(1, 3), help="MLP regression head: kernel of the first layer (1 = per voxel, 3 = 3x3x3 context).")
parser.add_argument("--cavity_prob", type=float, default=0.0,
                    help="Probability per DARK lesion that it lands at the local fluid level (a CSF-filled cavity, fluid pull 1.0) "
                         "instead of the tau-relative offset; other dark lesions keep --dark_fluid. 0 = off.")
parser.add_argument("--choroid_target", action="store_true",
                    help="Give the choroid plexus its own anatomy classes (57 left, 58 right, appended: 58 outputs). The target "
                         "voxels that are generation-label 9 inside a lateral ventricle are relabelled; the generator renders it as "
                         "before. Warm-starting a 56-class checkpoint copies its rows and initialises the two new ones.")
parser.add_argument("--freeze_bn_batch", action="store_true",
                    help="With --freeze_trunk: the frozen BatchNorm layers use the batch's own statistics (momentum 0) "
                         "instead of the running ones, matching the batch-statistics readout at inference.")
parser.add_argument("--bn_per_sample", action="store_true",
                    help="Every BatchNorm normalizes each batch element with its own statistics during training "
                         "(instance statistics, BatchNorm affine): the (clean, path) pair is not pooled, so training "
                         "matches the per-scan readout. Weights and checkpoints are unchanged; eval() keeps the buffers.")
parser.add_argument("--trunk_lr_mult", type=float, default=1.0,
                    help="Learning-rate multiplier for every parameter outside the regression head (two Adam groups: "
                         "trunk at 1e-4*M, head at 1e-4). 1.0 = one group as before; 0.1 = slow trunk under a warm head.")
parser.add_argument("--grad_balance", default="none", choices=("none", "imtl", "norm"),
                    help="Balance the SEG and REG gradients on the shared trunk every step (two backward passes). "
                         "imtl: impartial multi-task (Liu et al. 2021), two-task closed form, the combined trunk gradient "
                         "projects equally onto both task directions; norm: equal gradient norm. Scale-free, so neither "
                         "a fixed --seg_weight ratio nor the loss-value-based --uncertainty_weighting is needed. Heads "
                         "train on their own loss only.")
parser.add_argument("--grad_balance_every", type=int, default=1,
                    help="With --grad_balance imtl: recompute the two-pass balanced trunk gradient every K steps "
                         "and in between take ONE backward on ws*seg + wr*reg with the last weights, then divide each "
                         "head's gradient by its own weight (exact for the heads; the trunk uses a <=K-step-old alpha). "
                         "1 = every step (the original). The two backward passes were ~40%% of the step (2026-10-05).")
parser.add_argument("--lr_decay_from", type=int, default=0,
                    help="Cosine-decay the lr of every parameter group from this step to --max_steps, down to --lr_final_mult x "
                         "the construction lr. Assigned per step inside the loop, so it survives optimizer.load_state_dict on "
                         "--resume (which restores the checkpoint's lr). 0 = constant lr (the recipe).")
parser.add_argument("--lr_final_mult", type=float, default=0.1,
                    help="With --lr_decay_from: lr multiplier reached at --max_steps (0.1 = 1e-4 -> 1e-5).")
parser.add_argument("--ema", type=float, default=0.0,
                    help="Exponential moving average of the model weights, decay per step (0 = off; 0.9999 = 10k-step time "
                         "constant). Kept in every checkpoint as ema_state_dict (restored on --resume) and written as a sibling "
                         "synthseg_<step>_ema.pth whose model_state_dict holds the EMA weights, so compare_datasets.py infers "
                         "it unchanged (the batch readout ignores BN running statistics).")
parser.add_argument("--cudnn_benchmark", action="store_true",
                    help="torch.backends.cudnn.benchmark = True (autotune conv algorithms). OFF by default: it produced a CUDA "
                         "illegal-memory-access at step 0 in speed_bench 10168248 (2026-10-05); test before trusting")
parser.add_argument("--channels_last", action="store_true",
                    help="Keep the model and the training input in channels_last_3d (NDHWC) memory format: cuDNN's "
                         "tensor-core 3-D conv kernels; a layout change only, same arithmetic (speed lever, 2026-10-05)")
parser.add_argument("--compile", action="store_true",
                    help="torch.compile the training forward (elementwise fusion; shapes are fixed so no recompiles). "
                         "Checkpoints are saved from the uncompiled module, so keys are unchanged (speed lever, 2026-10-05)")
parser.add_argument("--freeze_trunk", action="store_true",
                    help="Train the regression head only: every other parameter frozen, the trunk's BatchNorm "
                         "in eval mode (running statistics fixed), optimizer over the head alone; the seg loss "
                         "is still computed but has no trainable parameter. Adam moments start fresh on resume.")
parser.add_argument("--atrophy_prob", type=float, default=None,
                    help="With --atrophy_coupled: probability that a draw carries secondary atrophy "
                         "(default: the generic coupling probability, 0.15).")
parser.add_argument("--init_encoder", type=str, default=None,
                    help="Initialise the encoder (enc1-4 + bottleneck) from a pretrained checkpoint, e.g. "
                         "pretrain_encoder.py's encoder_NNNNN.pth (motion-robust distillation). Decoder and "
                         "heads stay randomly initialised.")
parser.add_argument("--mode_weights", default=None,
                    help="Primary-mode categorical weights, 5 comma-separated floats in the generator's slot order (pathology_primitives "
                         "mode_names; the reported recipe is 0.23,0,0,0.385,0.385 under --no_diffuse). Overrides the --no_diffuse "
                         "default. 2026-10-26: mode-mix sensitivity fork (uniform 1/3 over the three live modes).")
parser.add_argument("--mode_polarity", action="store_true",
                    help="Mode-aware lesion polarity in BOTH intensity branches: on T1-like contrast vascular/"
                         "local/tract lesions are bright with p 0.10/0.25/0.15, on T2/FLAIR-like 0.85/0.85/0.95 "
                         "(default: floor branch uses the contrast prior, GMM branch a coin flip).")
parser.add_argument("--local_compartments", type=float, default=0.0,
                    help="generator.LOCAL_COMPARTMENT_PROB: with this probability a Local-Pathology lesion is rendered as nested core / rim / "
                         "edema compartments (edema grown through WM/GM and added to the lesion label; rim sign at a fraction of its deviation). 0 = off.")
parser.add_argument("--concentric_zones", action="store_true",
                    help="Ring lesions get intensity zones = core / shell (from the generator's ring geometry) "
                         "with independent signs, instead of spectral zones sharing one sign.")
parser.add_argument("--intra_lesion_gradient", type=float, default=0.0,
                    help="Lever 3: center->rim intensity ramp within the lesion (amplitude 0-1), for a "
                         "heterogeneous core-vs-rim rather than a flat lesion. 0 = off.")
parser.add_argument("--reg_tissue_weight", type=float, default=None,
                    help="With --reg_exclude_air: multiplier on healthy-tissue voxels in the regression "
                         "loss, under the legacy weights and normalisation with air's terms deleted. "
                         "1.0 = the control's loss minus air (tissue keeps its ~11%% of gradient); "
                         "~4.5 reproduces the 50/50 arm. Default (unset): 50/50 over the brain.")
parser.add_argument("--reg_exclude_air", action="store_true",
                    help="Drop background (seg label 0) from the REGRESSION loss entirely -- both the "
                         "contributions and the population the inverse-frequency balance is computed "
                         "over. Air is ~49%% of a volume and ~78%% of the zero pool, so it takes ~30%% "
                         "of regression gradient mass to fit a constant, while the hard negatives "
                         "(healthy tissue that looks lesion-like) get less. Measured: hard negatives "
                         "go 30%% -> 60%% of gradient mass at constant lesion share. Seg loss is "
                         "unaffected -- it still has to classify background.")
parser.add_argument("--gmm_mean_min", type=float, default=None,
                    help="Floor of the per-label GMM mean range (SynthSeg default 25 on 0..225). Lower it "
                         "so FLAIR-like draws can render CSF as dark as real nulled CSF; see generator.py.")
parser.add_argument("--gmm_sigma_rel", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="per-class GMM sigma = class mean x U[MIN,MAX] instead of the absolute U[5,25] "
                         "(generator.GMM_SIGMA_REL). Healthy-WM texture lever anchored on within-WM CV ~4-6%% "
                         "(Gispert 2004); e.g. --gmm_sigma_rel 0.02 0.08 (2026-10-05)")
parser.add_argument("--reg_quantile", type=float, default=0.5, metavar="Q",
                    help="pinball (asymmetric L1) regression loss toward the Q-quantile of the deviation, in the direction of the "
                         "target sign; 0.5 = the plain balanced L1 (2026-10-08). Clean rows and zero targets stay symmetric.")
parser.add_argument("--gmm_skew", type=float, default=0.0, metavar="A",
                    help="generator.GMM_SKEW: two-piece reshaping of the WM per-voxel GMM draw about the CSF direction "
                         "(heavy CSF-ward tail x(1+A|n|), light tail x(1-A)); pool-gated lever (2026-10-08)")
parser.add_argument("--noise_std", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="acquisition-grid noise std as a fraction of the WM mean, drawn U[MIN,MAX] per case; "
                         "generator default 0.05 0.15. Literature total within-WM CV of healthy T1w is ~0.06 "
                         "(Gispert 2004), so e.g. --noise_std 0.02 0.06 (2026-10-05)")
parser.add_argument("--healthy_hetero", type=float, default=0.0,
                    help="Max amplitude A of a smooth per-class intensity field on healthy WM/GM in both renderings "
                         "(amplitude ~ U(0, A) per image; target untouched). Teaches the head that mild normal-tissue "
                         "brightness variation is normal. 0 = off.")
parser.add_argument("--kseg_interp", type=float, default=0.0,
                    help="Probability per draw of rendering each healthy tissue (generation labels WM/GM/CSF) as k intensity "
                         "clusters of the subject's own T1w: 1-D k-means on the real intensities inside the label, k ~ U{--kseg_k}; "
                         "a cluster's mean is read off the piecewise-linear map through the real tissue medians and the drawn GMM "
                         "means, so real WM-ward texture lands WM-ward under any contrast. Only a label map is derived from the real "
                         "scan (no real intensity enters the image; independent of --mix_real_images); the same shift is added to "
                         "both renderings, so the regression target is unchanged. 0 = off (bit-identical).")
parser.add_argument("--kseg_k", type=int, nargs=2, default=[2, 8], metavar=("KMIN", "KMAX"),
                    help="Range of k for --kseg_interp, drawn uniformly per image.")
parser.add_argument("--kseg_contrasts", type=str, default="t1,t2,flair",
                    help="Which of the subject's real scans may enter the mixture --kseg_interp clusters (comma list of t1,t2,flair; "
                         "those on disk, random convex weights per image as in --mix_real_images, each standardized to WM=1; "
                         "mixtures that cancel a tissue gap are redrawn). T2w/FLAIR crops are read only on kseg draws.")
parser.add_argument("--contrast_t1", action="store_true",
                    help="Ablation: render a fixed T1-realistic contrast (WM>GM>CSF) instead of randomizing it, to specialize to T1 like the per-modality baselines. Detected/echoed at launch.")
parser.add_argument("--contrast_t2", action="store_true",
                    help="Ablation/gate: force a fixed T2-realistic contrast (CSF>GM>WM, CSF/WM 2-4) every image. For the in-domain T2w gate and a T2 specialization arm.")
parser.add_argument("--t2_prob", type=float, default=0.0,
                    help="Probability per image of drawing a T2-realistic contrast (unattenuated bright CSF, CSF>GM>WM). Enriches the rare realistic-T2w regime; 0 = off (legacy). Contrast-agnostic when <1.")
parser.add_argument("--csf_mix", type=float, default=0.0, metavar="P",
                    help="Per-draw probability of a second CSF intensity component (generator CSF_MIX_*: a fraction U[0.05,0.30] of "
                         "the CSF voxels, selected by a smooth field, rendered U[0.25,1] of the way from the CSF mean toward the GM mean; "
                         "literature-anchored, 2026-10-07). 0 = off.")
parser.add_argument("--dark_csf", type=float, default=0.0,
                    help="Probability per image that the CSF GMM mean is re-drawn as WM x U[0.1,0.4] (real T1/FLAIR "
                         "contrast); 0 = off (CSF mean independent of WM, the legacy prior).")
parser.add_argument("--norm_brain", type=float, default=0.0,
                    help="Probability per sample that the [p1,p99] normalization anchors are taken inside the brain "
                         "labels (the compare_datasets --norm_mask frame) instead of over all voxels > 0; 0 = off "
                         "(whole-foreground frame, the legacy behavior). Validation uses the brain frame whenever > 0.")
parser.add_argument("--dark_fluid", type=float, default=0.0,
                    help="Lever 4: pull DARK lesions this fraction of the way toward local fluid/CSF intensity "
                         "(0-1), matching chronic cavities/necrotic cores. 0 = off.")
parser.add_argument("--displace_clean", action="store_true",
                    help="Render the clean stream from the same displaced label maps as the "
                         "pathological one, so the pair differs only in the lesion and the "
                         "healthy segmentation target is aligned with both images.")
parser.add_argument("--clean_ref_smooth", type=float, default=None,
                    help="generator.LESION_CLEAN_REF_SMOOTH_SIGMA: blur (voxels) of the clean "
                         "reference the visibility correction rebuilds a subtle lesion from. "
                         "Measured 2026-09-13: synthetic lesion interiors are ~2x rougher than "
                         "real because that correction pushes the tissue's own voxel noise "
                         "through a per-voxel nonlinearity. Unset keeps the old behaviour.")
parser.add_argument("--extent_scale", type=str, default=None,
                    help="tract,vascular linear scale on the geometric lesion prior "
                         "(pathology_primitives.LESION_EXTENT_SCALE), e.g. 0.64,0.70. The "
                         "keep_frac route to a smaller lesion fragmented it into ~20 specks "
                         "(measured 2026-09-13) and collapsed WMH; this shrinks it as one blob. "
                         "Unset keeps the pre-fix geometry.")
parser.add_argument("--local_hemispheric", type=float, default=0.0,
                    help="pathology_primitives.LOCAL_HEMISPHERIC_PROB: probability that a Local lesion is placed "
                         "in cerebral WM/cortex rather than by volume share. 0 = off.")
parser.add_argument("--target_floor", type=float, default=0.0,
                    help="Regression-target magnitude below which a voxel is treated as "
                         "background by the balanced loss. Measured on validation_set_v2: 76%% "
                         "of the target's nonzero voxels are a rim with |target| <= 0.02, and "
                         "the inverse-frequency weight was giving them the full positive "
                         "weight. 0 keeps the exact-nonzero rule.")
parser.add_argument("--clean_weight", type=float, default=0.0,
                    help="Extra plain-L1 pull toward zero on the clean stream's regression "
                         "output. The balanced loss gives the whole zero population half the "
                         "gradient; this adds pressure where 'be silent' is unambiguous. "
                         "0 leaves the loss as it was.")
parser.add_argument("--reg_roi_weight", type=float, default=0.0,
                    help="Upweights the REGRESSION loss over the anatomical labels the injected "
                         "pathology overlaps, from the ground-truth healthy map. Trains the "
                         "head to separate lesion from the healthy remainder of its own "
                         "structure -- tissue-relative judgement taught through the loss, "
                         "with no dependence on the predicted segmentation at inference. "
                         "0 leaves the loss as it was.")
parser.add_argument("--roi_weight", type=float, default=0.0,
                    help="Upweights the healthy segmentation CE over the anatomical labels the "
                         "injected pathology overlaps, so the head learns to reconstruct "
                         "anatomy through a lesion. Spatial, not semantic: no tissue "
                         "identity is passed. 0 leaves the loss as it was.")
parser.add_argument("--uncertainty_weighting", action="store_true",
                    help="Use learned per-task log_sigma weighting instead of the fixed "
                         "--regression_weight. Off by default: sigma_reg collapses within a few "
                         "hundred steps and the regression term then dominates the shared trunk.")
parser.add_argument("--sigma_lr", type=float, default=1e-2,
                    help="Learning rate for the log_sigma_seg/log_sigma_reg parameter group (see "
                         "their own construction for the derivation and the simulation that chose "
                         "1e-2 -- converges in ~500-1000 steps there, versus ~20,000 sharing the "
                         "network's own 1e-4). That simulation held both task losses fixed to "
                         "isolate the weighting mechanism itself; it didn't model how fast the "
                         "resulting swing in relative task weight (precision_reg reached ~16x by "
                         "step 500 in one real run) might itself destabilize the network's shared "
                         "trunk before either task has a stable representation to weight in the "
                         "first place. If a run shows segmentation degrading or the regression "
                         "head producing widespread, poorly-localized output shortly after a rapid "
                         "sigma swing, a smaller value here (e.g. 1e-3 or 3e-3) trades slower "
                         "convergence to the balanced weighting for a gentler transition -- not "
                         "yet known which end of that trade-off is actually better, since this "
                         "flag is what makes it testable rather than fixed.")
parser.add_argument("--gen_device", type=str, default=None,
                    help="Run the generator on a separate CUDA device, e.g. cuda:1, with the "
                         "network on --device. The prefetch thread otherwise competes with "
                         "training for the same SMs and bandwidth. On two GPUs this cuts the "
                         "pathology step from ~0.44s to ~0.31s (roughly 1.4x, meaningful over a "
                         "500k-step run) and removes queue-wait almost entirely. Needs "
                         "--gres=gpu:2. Baseline (--do_pathology off) gains much less, since it's "
                         "only marginally generation-bound to begin with.")
parser.add_argument("--no_prefetch", dest="prefetch", action="store_false",
                    help="Disable background data generation. Prefetching overlaps generation "
                         "with the network pass; at pathology_prob=1.0 the two are close to "
                         "balanced (~0.35s vs ~0.38s) so this is worth ~1.9x. Ordering, "
                         "contents and RNG draws are identical either way.")
parser.add_argument("--susceptibility", action="store_true",
                    help="Model signal dropout from the lesion's own susceptibility, via the "
                         "dipole kernel in k-space (one FFT, a multiply, one inverse). Couples "
                         "appearance to lesion GEOMETRY rather than drawing it independently. "
                         "Fires on half of pathological draws so both the coupled and "
                         "independent regimes stay in support.")
parser.add_argument("--susceptibility_prob", type=float, default=1.0,
                    help="Fraction of pathological draws that get susceptibility dropout. "
                         "Defaults to 1.0: a lesion with susceptibility always shows the "
                         "effect, so whether a pathology has it at all is the flag's decision "
                         "rather than a per-draw coin flip.")
parser.add_argument("--coupling", action="store_true",
                    help="Derive lesion internal heterogeneity from lesion SIZE instead of "
                         "drawing it independently. Small lesions become homogeneous foci, "
                         "large ones get core/rim/surrounding zones -- a mass that outgrows "
                         "its blood supply necroses, which is biology rather than tuning. "
                         "Off by default; the generator otherwise treats content and "
                         "appearance as independent.")
parser.add_argument("--complex_resample", action="store_true",
                    help="Model low resolution by cropping k-space rather than blurring in image "
                         "space. Physically what a low-res acquisition does, and produces real "
                         "Gibbs ringing instead of interpolation banding. Off by default.")
parser.add_argument("--restricted_resolution_sampler", action="store_true",
                    help="Use an always-anisotropic resolution sampler (one axis at 2.5-8.5mm, "
                         "other two fixed at 1mm) instead of SynthSeg's, which is isotropic "
                         "U(1,4mm) half the time and anisotropic U(1,8mm) on one axis the other "
                         "half.")
parser.add_argument("--mix_real_images", action="store_true",
                    help="Blend the subject's real T1/T2/FLAIR into half the synthetic training "
                         "images. Off by default: SynthSeg's training data contains no real scans, "
                         "and enabling this breaks that claim.")
parser.add_argument("--image_source", choices=["synthetic", "real"], default="synthetic",
                    help="Data-kind control arm (2026-10-20): 'real' = the clean image is the subject's REAL scan (one contrast per "
                         "draw, uniform over its T1w/T2w/FLAIR on disk), the pathological image the same scan plus the drawn anomaly; "
                         "HR stage down unchanged; dark_csf/csf_mix/healthy_hetero skipped.")
parser.add_argument("--anomaly_source", choices=["prior", "fpi"], default="prior",
                    help="'prior' = our label-space prior (with --image_source real: painted into the real scan through the "
                         "inverse kseg anchor map, FAE-kind); 'fpi' = DISYRE's foreign-patch blend + v2 codebook bias from a "
                         "same-contrast donor (DISYRE-kind; needs --image_source real).")
parser.add_argument("--fpi_shapes", type=str, default=os.environ.get('PR_FPI_SHAPES', '/path/to/fpi_shapes'),
                    help="DISYRE's 300 binary shape volumes (dataset.json) for --anomaly_source fpi; 'none' = box/sphere only.")
parser.add_argument("--fpi_patch", type=int, default=64, help="--anomaly_source fpi: DISYRE's anom_patch_size (voxels, cubic).")
parser.add_argument("--fpi_raw", action="store_true",
                    help="--anomaly_source fpi: skip the Rose visibility floor/ceiling on the DISYRE label, keeping its alpha ~ "
                         "U(0,1) severities as drawn (default: the correction runs as on every lesion, re-grading them).")
parser.add_argument("--bf16", action="store_true",
                    help="autocast in bfloat16 and run the GradScaler disabled (bf16 has fp32's exponent range, so no loss "
                         "scaling). 2026-10-26: under fp16 the real-image + FPI arm's SEG trunk gradient (CE+Dice averaged over "
                         "~4M voxels, ~1e-7 per voxel) underflowed to exactly 0 once recurring backward overflows pinned the "
                         "scaler at 1e-3..1e-6 (checkpoint scaler_state_dict), freezing the seg head from step 12.7k.")
parser.add_argument("--healthy", action="store_true", help="Restrict training to healthy subjects only (HCP, Chinese-HCP, and subsets of ADNI3/OASIS3).")
parser.add_argument("--data_dir", type=str, default=os.environ.get('PR_DATA_DIR', '/path/to/data_dir') + '/',
                     help="Training data directory. Was a hardcoded path with no override -- if it ever gets renamed again, pass the new one here instead of editing this file.")
parser.add_argument("--validation_dir", type=str, default=None,
                     help="Directory of val_case_NNNN.pt files from pregenerate_validation_set.py. "
                          "If given, this fixed set is evaluated every --validation_every steps -- "
                          "the epistemic-uncertainty check: same cases every time, so a checkpoint "
                          "changing score reflects the model, not which random case got drawn.")
parser.add_argument("--validation_every", type=int, default=10000,
                     help="How often (in steps) to run the fixed validation set. Ignored if "
                          "--validation_dir isn't given.")
parser.add_argument("--validation_batch_size", type=int, default=1,
                     help="How many validation cases to batch per forward pass (each becomes "
                          "network batch 2*N: clean+pathological). Default 1 (network batch 2) "
                          "matches what the periodic training-log diagnostic already runs safely "
                          "every 500 steps -- the model's own weights and optimizer state are "
                          "already resident on this GPU throughout, unlike a standalone eval "
                          "script with the whole device to itself. A batch of N validation cases "
                          "at full volume resolution allocates a (2N,56,160,160,160) float32 "
                          "tensor for the softmax alone, so headroom shrinks fast as N grows. "
                          "Raise this only if you've confirmed real headroom on your specific "
                          "GPU/resolution combination.")
args = parser.parse_args()
if args.mode_weights:
    _mw = [float(v) for v in args.mode_weights.split(",")]
    if len(_mw) != 5 or abs(sum(_mw) - 1.0) > 1e-3: raise SystemExit(f"--mode_weights needs 5 values summing to 1, got {_mw}")
    if (args.no_diffuse or args.atrophy_coupled) and (_mw[1] != 0.0 or _mw[2] != 0.0):
        raise SystemExit("--mode_weights slots 2-3 (diffuse, atrophy) must be 0 under --no_diffuse / --atrophy_coupled")
if args.no_tissue_rel:
    args.tissue_rel = False
    print("no_tissue_rel: own-class channel OFF (ablation); regression head reads the decoder features only", flush=True)
if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
if args.no_diffuse or args.atrophy_coupled:
    pathology_primitives.DISABLED_MODES = {"Global Pathology"}
    if not args.atrophy_coupled:
        pathology_primitives.DISABLED_MODES.add("Geometric Atrophy")
if args.atrophy_coupled:
    pathology_primitives.ATROPHY_VENTRICLE_PROB = 0.5
if args.atrophy_prob is not None:
    pathology_primitives.COUPLE_PROB_BY_MODE["Geometric Atrophy"] = args.atrophy_prob
if args.displace_clean:
    import generator as _generator_mod
    _generator_mod.DISPLACE_CLEAN = True
if args.adni or args.oasis:
    import generator as _generator_mod
    _generator_mod.HEALTHY_EXTRA_DATASETS = tuple(d for d, on in (('ADNI3', args.adni), ('OASIS3', args.oasis)) if on)
    _generator_mod.OASIS3_EXCLUDE_FILE = args.oasis_exclude
if args.buckner40:
    import generator as _generator_mod
    _generator_mod.ALLOW_BUCKNER40 = True
    print("buckner40 on: Buckner40 admitted UNFILTERED (no diagnosis table; contains demented subjects)", flush=True)
if args.exclude_datasets:
    import generator as _generator_mod
    _generator_mod.EXCLUDE_DATASETS = tuple(args.exclude_datasets)
if args.concentric_zones:
    import generator as _generator_mod
    _generator_mod.CONCENTRIC_ZONES = True
if args.local_compartments > 0:
    import generator as _generator_mod
    _generator_mod.LOCAL_COMPARTMENT_PROB = args.local_compartments
    print(f"local_compartments on (p={args.local_compartments})", flush=True)
if args.mode_polarity:
    import generator as _generator_mod
    _generator_mod.POLARITY_BY_MODE = {"vascular": {"t1": 0.10, "t2": 0.85}, "local": {"t1": 0.25, "t2": 0.85},
                                       "tract": {"t1": 0.15, "t2": 0.95}}
if args.tissue_sub:
    import generator as _generator_mod
    _generator_mod.TISSUE_SUB = {k: float(v) for k, v in (kv.split(":") for kv in args.tissue_sub.split(","))}
if args.intra_lesion_gradient > 0:
    import generator as _generator_mod
    _generator_mod.INTRA_LESION_GRADIENT = args.intra_lesion_gradient
if args.cavity_prob > 0:
    import generator as _generator_mod
    _generator_mod.CAVITY_PROB = args.cavity_prob
    print(f"cavity_prob {args.cavity_prob}", flush=True)
if args.dark_fluid > 0:
    import generator as _generator_mod
    _generator_mod.DARK_FLUID = args.dark_fluid
if args.gmm_mean_min is not None:
    import generator as _generator_mod
    _generator_mod.GMM_MEAN_MIN = args.gmm_mean_min
if args.healthy_hetero > 0:
    import generator as _generator_mod
    _generator_mod.HEALTHY_HETERO = args.healthy_hetero
    print(f"healthy_hetero A={args.healthy_hetero}: smooth per-class intensity fields on healthy WM/GM (both renderings)", flush=True)
if args.kseg_interp > 0:
    import generator as _generator_mod
    assert 0 < args.kseg_interp <= 1 and 1 <= args.kseg_k[0] <= args.kseg_k[1], "--kseg_interp in (0,1], --kseg_k KMIN<=KMAX"
    _generator_mod.KSEG_PROB = args.kseg_interp
    _generator_mod.KSEG_K = (args.kseg_k[0], args.kseg_k[1])
    _kc = tuple(c.strip().lower() for c in args.kseg_contrasts.split(",") if c.strip())
    assert _kc and set(_kc) <= {"t1", "t2", "flair"}, "--kseg_contrasts: comma list of t1,t2,flair"
    _generator_mod.KSEG_CONTRASTS = _kc
    print(f"kseg_interp p={args.kseg_interp} k in [{args.kseg_k[0]},{args.kseg_k[1]}] classes {_generator_mod.KSEG_CLASSES} contrasts {_kc}: "
          f"healthy WM/GM/CSF rendered as k-means clusters of a random mixture of the subject's real scans, means on the real->GMM anchor map (both renderings)", flush=True)
def _patch_module_global(mod, name, value):
    # hasattr guard: a module without the knob warns instead of taking a dead attribute
    if hasattr(mod, name):
        setattr(mod, name, value)
        return True
    print(f"WARNING: {mod.__name__}.{name} does not exist (stale module?); {name}={value} NOT applied", flush=True)
    return False


if args.gmm_sigma_rel is not None:
    import generator as _generator_mod
    _lo, _hi = float(args.gmm_sigma_rel[0]), float(args.gmm_sigma_rel[1])
    if not (0.0 < _lo <= _hi): raise SystemExit(f"--gmm_sigma_rel needs 0 < MIN <= MAX, got {args.gmm_sigma_rel}")
    if _patch_module_global(_generator_mod, "GMM_SIGMA_REL", (_lo, _hi)):
        print(f"[generator] per-class GMM sigma = mean x U[{_lo}, {_hi}] (default absolute U[5,25])", flush=True)
if args.gmm_skew > 0:
    import generator as _generator_mod
    if not (0.0 < args.gmm_skew < 1.0): raise SystemExit(f"--gmm_skew needs 0 < A < 1, got {args.gmm_skew}")
    if _patch_module_global(_generator_mod, "GMM_SKEW", float(args.gmm_skew)):
        print(f"[generator] gmm_skew on (a={args.gmm_skew}): WM draw heavy CSF-ward, light otherwise", flush=True)
if args.noise_std is not None:
    import generator as _generator_mod
    _lo, _hi = float(args.noise_std[0]), float(args.noise_std[1])
    if not (0.0 <= _lo <= _hi): raise SystemExit(f"--noise_std needs 0 <= MIN <= MAX, got {args.noise_std}")
    if _patch_module_global(_generator_mod, "NOISE_STD_MIN", _lo) and _patch_module_global(_generator_mod, "NOISE_STD_MAX", _hi):
        print(f"[generator] acquisition noise std U[{_lo}, {_hi}] of WM (default U[0.05, 0.15])", flush=True)

if args.contrast_t1:
    import generator as _generator_mod
    if _patch_module_global(_generator_mod, "CONTRAST_T1", True):
        print("contrast_t1 on (T1-realistic contrast; specialization ablation)", flush=True)

if args.contrast_t2:
    import generator as _generator_mod
    if _patch_module_global(_generator_mod, "CONTRAST_T2", True):
        print("contrast_t2 on (T2-realistic contrast: CSF>GM>WM; forced every image)", flush=True)

if args.t2_prob > 0:
    import generator as _generator_mod
    _generator_mod.T2_CONTRAST_PROB = args.t2_prob
    print(f"t2_prob={args.t2_prob}: T2-realistic contrast (bright unattenuated CSF) drawn per image at this rate", flush=True)

if args.csf_mix > 0:
    _generator_mod.CSF_MIX_PROB = args.csf_mix
    print(f"csf_mix on (p={args.csf_mix})", flush=True)
if args.dark_csf > 0:
    import generator as _generator_mod
    _generator_mod.DARK_CSF_PROB = args.dark_csf
    print(f"dark_csf on (p={args.dark_csf})", flush=True)
if args.anomaly_source == "fpi" and args.image_source != "real":
    raise SystemExit("--anomaly_source fpi needs --image_source real (DISYRE's anomaly is a patch of a real scan)")
if args.image_source == "real":
    import generator as _generator_mod
    if args.mix_real_images:
        raise SystemExit("--image_source real already uses the real scan; drop --mix_real_images")
    _generator_mod.IMAGE_SOURCE = "real"
    print("image_source real: clean = the subject's real scan (contrast uniform over its T1w/T2w/FLAIR on disk), pathological = "
          "the same scan with the drawn anomaly; the synthetic rendering is used only for the lesion paint; "
          "dark_csf/csf_mix/healthy_hetero skipped (rendering-only levers)", flush=True)
if args.anomaly_source == "fpi":
    _generator_mod.ANOMALY_SOURCE = "fpi"
    _generator_mod.FPI_PATCH = int(args.fpi_patch)
    if args.fpi_shapes.lower() == "none":
        _generator_mod.FPI_SHAPES_DIR = None
        print("fpi_shapes none: DISYRE shape kinds box and sphere only (no shape volumes)", flush=True)
    elif os.path.isfile(os.path.join(args.fpi_shapes, "dataset.json")):
        _generator_mod.FPI_SHAPES_DIR = args.fpi_shapes
    else:
        raise SystemExit(f"--fpi_shapes {args.fpi_shapes}: no dataset.json (DISYRE's shape volumes); pass 'none' for box/sphere only")
    if args.fpi_raw:
        _generator_mod.FPI_VISIBILITY = False
    print(f"anomaly_source fpi: DISYRE foreign-patch blend + v2 codebook bias from a same-contrast donor (patch {args.fpi_patch}^3, "
          f"alpha ~ U(0,1), shapes {_generator_mod.FPI_SHAPES_DIR}); prior injection and mass effect off; "
          f"visibility correction {'OFF (--fpi_raw)' if args.fpi_raw else 'on'}", flush=True)
elif args.fpi_raw:
    raise SystemExit("--fpi_raw needs --anomaly_source fpi")
if args.attention_dense and not args.attention:
    raise SystemExit("--attention_dense needs --attention")
if args.attention_dense:
    print("attention_dense on (att1/2/3 also gated by every coarser decoder level)", flush=True)
if args.clean_ref_smooth is not None:
    import generator as _generator_mod
    _generator_mod.LESION_CLEAN_REF_SMOOTH_SIGMA = args.clean_ref_smooth
if args.extent_scale is not None:
    _t, _v = (float(x) for x in args.extent_scale.split(","))
    pathology_primitives.LESION_EXTENT_SCALE = {"tract": _t, "vascular": _v}


if args.lesion_comp:
    import generator as _generator_mod
    _lc = {k: float(v) for k, v in (kv.split(":") for kv in args.lesion_comp.split(","))}
    _bad = sorted(set(_lc) - {"vascular", "local", "tract"})
    if _bad:
        raise SystemExit(f"--lesion_comp: unknown mode(s) {_bad}; keys are vascular|local|tract")
    if _patch_module_global(_generator_mod, "LESION_COMPOSITION", _lc):
        print(f"lesion_comp {_lc}", flush=True)
if args.lesion_comp_f:
    import generator as _generator_mod
    _lo, _hi = (float(v) for v in args.lesion_comp_f.split(","))
    if not (0.0 < _lo <= _hi):
        raise SystemExit(f"--lesion_comp_f: need 0 < LO <= HI, got {args.lesion_comp_f}")
    if _patch_module_global(_generator_mod, "COMP_F_CSF", (_lo, _hi)):
        print(f"lesion_comp_f csf zones f in [{_lo}, {_hi}]", flush=True)
if args.comp_min_k is not None:
    import generator as _generator_mod
    if _patch_module_global(_generator_mod, "COMP_MIN_K", float(args.comp_min_k)):
        print(f"comp_min_k {args.comp_min_k}", flush=True)
if args.local_hemispheric > 0:
    if _patch_module_global(pathology_primitives, "LOCAL_HEMISPHERIC_PROB", args.local_hemispheric):
        print(f"local_hemispheric P={args.local_hemispheric}", flush=True)
if args.target_region > 0:
    if args.target_floor >= args.target_region:
        raise SystemExit(f"--target_region {args.target_region} must exceed --target_floor {args.target_floor}, "
                         "else the floored interior counts as background in the balanced loss")
    import generator as _generator_mod
    # keep the geometric label; its sub-tau voxels get sign x C in the target
    _patch_module_global(_generator_mod, "LABEL_SUPPRESS_INVISIBLE", False)
    print(f"target_region C={args.target_region}: lesion interior floored to sign x C (LABEL_SUPPRESS_INVISIBLE=False)", flush=True)
if args.task == "register" and (args.reg_head != "linear" or args.tissue_rel or args.freeze_trunk):
    raise SystemExit("--reg_head/--tissue_rel/--freeze_trunk need the regression head (--task segment)")
if args.reg_head != "linear":
    print(f"reg_head {args.reg_head}", flush=True)
if args.tissue_rel:
    print("tissue_rel on" + (f" tissue_refs {args.tissue_refs}" if args.tissue_refs else "") + (f" tissue_vec {args.tissue_vec}" if args.tissue_vec != "none" else "")
          + (f" tissue_pool {args.tissue_pool}" if args.tissue_pool != "own" else "") + (f" tissue_unroll {args.tissue_unroll}" if args.tissue_unroll else ""), flush=True)
if args.reg_head == "mlp" and (args.reg_hidden, args.reg_depth, args.reg_kernel) != (32, 2, 1):
    print(f"reg_arch hidden {args.reg_hidden} depth {args.reg_depth} kernel {args.reg_kernel}", flush=True)
if args.tissue_unroll and not args.tissue_rel:
    raise SystemExit("--tissue_unroll needs --tissue_rel")
if args.tissue_vec != "none" and not args.tissue_rel:
    raise SystemExit("--tissue_vec needs --tissue_rel")
if args.tissue_pool != "own" and not args.tissue_rel:
    raise SystemExit("--tissue_pool needs --tissue_rel")
if args.tissue_soft and (not args.tissue_rel or args.tissue_pool != "own"):
    raise SystemExit("--tissue_soft needs --tissue_rel and --tissue_pool own")
if args.tissue_soft:
    print("tissue_soft on", flush=True)
if args.freeze_bn_batch:
    if not args.freeze_trunk:
        raise SystemExit("--freeze_bn_batch needs --freeze_trunk")
    print("freeze_bn_batch on", flush=True)

do_motion = args.do_motion
do_pathology = args.do_pathology
resume = args.resume

# --- 1. Setup Directories & Logging ---
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

# Dynamically name the run and output folder based on which augmentations are
# active. The git hash goes on EVERY run, not just pathology ones: each run is
# launched from its own worktree pinned to a commit, so the hash is what ties a
# checkpoint directory back to the code that produced it. Attaching it only to
# the pathology branch meant a baseline run landed in a bare "healthy_out",
# indistinguishable from every other baseline run at any commit.
# (sbatch --job-name only names the SLURM job; it never reaches this script.)
run_prefix_parts = [p for p, on in [("motion", do_motion),
                                     ("mcons", args.motion_consistency),
                                     ("complex_resample", args.complex_resample),
                                     ("susc", args.susceptibility),
                                     ("coupled", args.coupling),
                                     ("pathology", do_pathology),
                                     ("healthy", args.healthy)] if on]
run_prefix_parts = run_prefix_parts or ["baseline"]
run_prefix = "_".join(run_prefix_parts + ([args.tag] if args.tag else []) + [_git_identifier()])

# ---------------------------------------------------------
# FIXED: Route all saves to space_dir by joining it to root_dir
# ---------------------------------------------------------
space_dir = os.environ.get('PR_OUT_DIR', '/path/to/out_dir') + '/'
root_dir = os.path.join(space_dir, f"{run_prefix}_out")

resume_ckpt_path = os.environ.get('PR_OUT_DIR', '/path/to/out_dir') + '/'
start_step = 0
target_steps = 500000
if args.max_steps:
    # 2026-10-05: may now RAISE the horizon as well as lower it. Nothing else is tied to target_steps (no
    # schedule reads it; it names the run and bounds the loop), and the recipe's benchmark legs were still
    # improving at 500k (BraTS T2w +0.026/100k, ATLAS-O +0.015/100k, both seeds up on T2w).
    target_steps = int(args.max_steps)

if resume:
    # Find the most recent existing run directory for this mode
    # FIXED: Now searches inside root_dir (which is inside space_dir)
    existing_runs = sorted(glob.glob(os.path.join(root_dir, "log_*")))
    if not existing_runs:
        raise FileNotFoundError(
            f"--resume was passed but no existing run directories were found under '{root_dir}'."
        )
    log_dir = existing_runs[-1]

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    progress_dir = os.path.join(log_dir, "progress")

    existing_ckpts = glob.glob(os.path.join(ckpt_dir, "synthseg_*.pth"))
    if not existing_ckpts:
        raise FileNotFoundError(
            f"--resume was passed but no checkpoints were found in '{ckpt_dir}'."
        )

    resume_ckpt_path = max(
        existing_ckpts,
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
    )
    start_step = int(os.path.basename(resume_ckpt_path).split("_")[1].split(".")[0])

    # Reuse the original timestamp so RUN_NAME reflects the run being resumed
    timestamp = os.path.basename(log_dir).replace("log_", "")
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(root_dir, f"log_{timestamp}")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    progress_dir = os.path.join(log_dir, "progress")

RUN_NAME = f"{run_prefix}_{target_steps}_{timestamp}"

# os.makedirs automatically creates parent directories (like root_dir and space_dir) if they don't exist
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(progress_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "train_log.txt")
log_file = open(log_file_path, "a")

def log_print(msg):
    t = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{t}] {msg}"
    print(formatted_msg)
    log_file.write(formatted_msg + "\n")
    log_file.flush()

log_print(f"Initialized run. Logging to: {log_dir}")
# Every flag, not a hand-picked subset: a checkpoint directory is named by
# commit hash and prefix, neither of which captures probabilities or the
# flags that do not appear in the prefix. Without this the log cannot tell
# you what a run actually did.
log_print("Full configuration:")
for _k, _v in sorted(vars(args).items()):
    log_print(f"    {_k:34s} {_v}")
log_print(f"Motion Augmentation Enabled: {do_motion}" + (f" (Prob: {args.motion_prob})" if do_motion else ""))
log_print(f"Pathology Augmentation Enabled: {do_pathology}" + (f" (Prob: {args.pathology_prob}) -- pathology is always generated by paired_pathology_generator; this flag/prob controls whether it's used in the loss" if do_pathology else ""))
log_print(f"Healthy Filter Enabled: {args.healthy}")
if args.norm_brain > 0:
    log_print(f"norm_brain P={args.norm_brain}: brain-frame percentiles on a random P of samples (validation: always)")
log_print(f"Resume: {resume}" + (f" (from {resume_ckpt_path}, step {start_step})" if resume else ""))

training_data_dir = args.data_dir
target_device = 'cuda'
# CUDA sync is required for the per-step timers to mean anything; see the loop.
_timing = torch.cuda.is_available() and target_device == 'cuda'

LABEL_LIST_SEGMENTATION = [0, 11, 12, 13, 16, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 17, 47, 49, 51, 53, 55, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 48, 50, 52, 54, 56]
if args.choroid_target:
    LABEL_LIST_SEGMENTATION = LABEL_LIST_SEGMENTATION + [57, 58]   # compact 56 = left, 57 = right choroid plexus
    import generator as _generator_mod
    _generator_mod.CHOROID_TARGET = True
    print(f"choroid_target on ({len(LABEL_LIST_SEGMENTATION)} classes)", flush=True)
LESION_COMPACT_IDX = LABEL_LIST_SEGMENTATION.index(33) if 33 in LABEL_LIST_SEGMENTATION else None
HALO_RADIUS = 5

# Built once here, from the SAME list the generator uses below, so its
# indexing (compact index = position in LABEL_LIST_SEGMENTATION) always
# matches what argmax(...) over the generator's one-hot targets and the
# model's own predictions are already indexed by.
FS_CMAP, FS_N_LABELS = build_fs_colormap(LABEL_LIST_SEGMENTATION)


# --- Dynamic Curriculum for the Rose Criterion ---
# Shared state so the generator can query the current step
train_state = {'step': start_step}

def k_rose_sampler():
    """Contrast target, uniform on [1.0, 2.5] surround-sigma.

    Referenced to the surrounding tissue's own variation, not sensor noise:
    k = 1 sits at the edge of a neighbourhood's normal variation, k = 5 is
    conspicuous. Real pathology spans both.

    Uniform because coverage is won at the extremes. The upper bound is a
    chi-vs-R_syn trade: too high and conspicuous synthetic lesions are learned
    easily while subtle real ones fall outside the synthetic support.
    """
    return float(np.random.uniform(1.0, 2.5))

# Fingerprints the LIVE k_rose_sampler function (not its source text) into
# this run's log, so "which range did this run actually use" is answered by
# the log header from now on instead of having to be reconstructed later.
diag.report_sampler_provenance(
    k_rose_sampler,
    expected_range=(1.0, 2.5),
    git_hash=_git_identifier(),
    log_print=log_print,
)
# Accumulates (requested_k, realized_k) pairs and normalization-cascade
# ratios across the whole run; see the [viz step] block and end-of-training
# summary below.
RUNNING_DIAG = diag.RollingDiagnostics()

# Initialize generator. Pathology (via PathologyRandomizer) is injected into
# the label maps BEFORE GMM synthesis and the shared deformation/bias-field/
# resolution/noise pipeline, and every downstream random draw is shared
# between the clean and pathological pass -- so this always yields a
# genuinely controlled pair, not two independently-augmented cases.
mode_sampler = make_mode_sampler(restricted=args.restricted_resolution_sampler)

gen = paired_pathology_generator(
    training_data_dir, 
    mode_sampler, 
    label_list_segmentation=LABEL_LIST_SEGMENTATION,
    n_neutral_labels=20, 
    label_list_normalization=[1, 18], 
    sides='both', 
    labels_to_kill=[45],  
    list_to_kill_photo=[],
    list_to_kill_photo_inc_ventricles=[],
    flipping=True, 
    # SynthSeg complements its segmentations with extra-cerebral labels and
    # notes "these new labels are dropped with 50% chances during
    # generation, to make SynthSeg compatible with skull stripped images".
    # Dropped as a group, all-or-nothing -- hence frac=(1.0, 1.0).
    mix_real_images=args.mix_real_images,
    susceptibility=args.susceptibility,
    couple_appearance=args.coupling,
    susceptibility_prob=args.susceptibility_prob,
    complex_resample=args.complex_resample,
    # Tukey apodization only makes sense alongside k-space truncation.
    gibbs_ringing_filter=args.complex_resample,
    # Don't build the pathological stream at all when the training loop will
    # not use it -- see generate_pathology in generator.py.
    generate_pathology=do_pathology,
    # Skip loading the 15 of 18 volumes per case that nothing consumes.
    minimal_io=True,
    # Yield argmax indices rather than the 56-channel one-hot: this loop
    # argmaxes immediately anyway, and the one-hot is 0.92 GB per stream at
    # 160^3 against 0.033 GB for indices. Prefetching holds cases in a queue,
    # and at depth 2 the one-hot form kept 3.74 GB of GPU memory alive, which
    # starved cuFFT inside the generator (reported as CUFFT_INVALID_SIZE).
    compact_labels=True,
    random_kill_labels=list(range(34, 47)),
    random_kill_prob=0.5,
    random_kill_frac=(1.0, 1.0),
    # Generator device is separate from the network's when --gen_device is
    # given; cases are moved across in the training loop below.
    device=(args.gen_device or target_device),
    healthy_only=args.healthy,
    k_rose_sampler=k_rose_sampler,
    # --no_diffuse / --atrophy_coupled: zero the primary draw here; the
    # coupling roll is masked in pathology_primitives via DISABLED_MODES above.
    **({"mode_weights": tuple(float(v) for v in args.mode_weights.split(","))} if args.mode_weights else
       ({"mode_weights": (0.23, 0.0, 0.0, 0.385, 0.385)} if (args.no_diffuse or args.atrophy_coupled) else {})),
)

if args.prefetch and not resume:
    # Produce the next case on a background thread while the current one
    # trains, so a step costs max(gen, net) rather than gen + net. A thread
    # rather than a process: the generator already runs on the GPU, and its
    # time goes into torch kernels and numpy, both of which release the GIL.
    # On --resume the wrap waits for the checkpoint load, so the restored RNG
    # stream feeds every draw.
    gen = Prefetcher(gen, depth=2)
    log_print("Prefetch enabled (background generation, depth 2).")
if args.gen_device:
    log_print(f"Generator on {args.gen_device}, network on {target_device}. "
              "Cases are copied across per step; with compact_labels that is "
              "~8 MB, small against what the separation saves.")

# --- 2. The Stable SynthSeg U-Net Architecture ---
# SynthSeg builds its UNet with batch_norm=-1, i.e. BatchNormalization on the
# channel axis. At batch size 1 BatchNorm and InstanceNorm behave identically
# during TRAINING (both normalize over spatial dims per channel), but they
# diverge at inference: BatchNorm switches to stored running statistics, while
# InstanceNorm keeps recomputing from whatever it is fed. That makes
# InstanceNorm inference sensitive to how much of the input is air, which is
# why whole-volume inference on a large mostly-empty FOV misbehaves.
#
# NOTE: switching norm changes the state_dict keys (BatchNorm carries
# running_mean/running_var/num_batches_tracked). Checkpoints trained with one
# cannot be --resume'd with the other.



# --- 3. Stable Combined Loss Function ---
class SoftDiceLoss(nn.Module):
    """Dice averaged over foreground classes, with an optional per-class
    weight (lesion_idx, lesion_weight) for a rare foreground class that an
    unweighted mean would otherwise barely penalize.

    A weighted mean rather than a scaled extra term, so the loss stays
    bounded in [0, 1] and the weight's meaning is legible: w = 1 is the
    unweighted default, larger w makes that class count for more of the
    foreground term. A large weight trades anatomy accuracy for sensitivity
    on that class, producing false positives in tissue the network would
    otherwise label correctly.

    Currently unused: the classification target no longer contains a Lesion
    class (see the training loop's target construction), so lesion_idx is
    always None and this reduces to a plain unweighted mean.
    """

    def __init__(self, smooth=1.0, lesion_idx=None, lesion_weight=1.0):
        super().__init__()
        self.smooth = smooth
        self.lesion_idx = lesion_idx
        self.lesion_weight = lesion_weight

    def forward(self, logits, targets_idx, mask=None):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets_idx, num_classes).permute(0, 4, 1, 2, 3).float()
        if mask is not None:
            # excluded voxels (mask 0) drop out of both intersection and denominator
            m = mask.to(probs.dtype).unsqueeze(1)
            probs = probs * m
            targets_one_hot = targets_one_hot * m

        probs = probs.view(probs.size(0), num_classes, -1)
        targets = targets_one_hot.view(targets_one_hot.size(0), num_classes, -1)

        intersection = (probs * targets).sum(-1)
        den = probs.sum(-1) + targets.sum(-1)
        dice_per_class = (2. * intersection + self.smooth) / (den + self.smooth)

        fg = dice_per_class[:, 1:]                      # drop background
        if self.lesion_idx is None or self.lesion_weight == 1.0:
            return 1 - fg.mean()

        w = torch.ones(fg.shape[1], device=fg.device, dtype=fg.dtype)
        w[self.lesion_idx - 1] = self.lesion_weight     # -1: background dropped
        return 1 - (fg * w).sum(1).div(w.sum()).mean()


class WeightedRegressionLoss(nn.Module):
    """L1 loss with inverse-frequency weighting on genuinely-pathological
    voxels, addressing the same class-imbalance problem lesion_weight above
    used to address for the classification head, now for the regression
    head instead.

    Verified directly: at realistic sparsity (~0.2% of a volume genuinely
    pathological), plain unweighted L1 scores a USELESS prediction that
    collapses to near-zero everywhere almost 12x LOWER loss than a
    genuinely correct one -- the vast always-zero background dominates the
    average, so gradient descent has every incentive to learn "predict
    zero" and ignore the sparse signal entirely, regardless of how long
    training runs.

    No threshold, no max_weight -- both were arbitrary hand-picked
    constants, replaced here by principled alternatives rather than
    better-guessed numbers:

    "Genuinely nonzero" is target != 0, exact rather than approximate.
    Untouched voxels are exactly zero under the generator's shared-noise
    design, not approximately -- and tanh(0) = 0 exactly, so that property
    survives the regression head's own bounding unchanged. No cutoff needs
    guessing when the true boundary is exact.

    The weight itself is uncapped inverse frequency (n_zero / n_nonzero),
    computed PER BATCH from the actual observed sparsity, not a fixed
    constant -- lesion size varies enormously across modes, and a fixed
    weight tuned for one size is off by orders of magnitude for another.
    Uncapped is safe because the loss's own normalization (dividing by
    weight.sum()) already bounds the ACTUAL per-voxel gradient
    contribution regardless of how large the raw weight gets: verified
    directly, a single-voxel lesion in a 4-million-voxel volume computes a
    raw weight near 4 million, but the realized per-voxel gradient after
    normalization is a well-behaved 0.5, not a blowup. The raw weight
    achieving exact 50/50 population balance was never the risk; a
    hand-picked ceiling was solving a problem the normalization had
    already solved.
    """

    def __init__(self, floor=0.0, tissue_weight=None, quantile=0.5):
        super().__init__()
        self.floor = floor
        # quantile != 0.5: pinball (asymmetric L1) in the direction of the target's sign, so the head reports the
        # q-quantile of the deviation posterior instead of its median -- nonzero wherever the lesion posterior
        # exceeds 1 - q instead of 1/2 (the hedge measured 2026-10-08: residual 0.01-0.09 on missed T2w/ATLAS
        # voxels vs 0.29 on hits). Scaled 2q / 2(1-q) so q = 0.5 is exactly the plain L1. Zero targets stay symmetric.
        self.quantile = float(quantile)
        # Only used with mask=. None: 50/50 lesion/tissue over the masked
        # population. A float tau: the legacy per-voxel weights and
        # normalization (air counted in the denominator), air's contribution
        # deleted, healthy tissue scaled by tau; tau = 1 is the control's
        # loss minus its air terms.
        self.tissue_weight = tissue_weight

    def _err(self, pred, target):
        e = (pred - target).abs()
        q = self.quantile
        if q == 0.5:
            return e
        under = (pred - target) * torch.sign(target) < 0          # short of the target in its own direction
        scale = torch.where(target == 0, torch.ones_like(e), torch.where(under, torch.full_like(e, 2 * q), torch.full_like(e, 2 * (1 - q))))
        return scale * e

    def forward(self, pred, target, weight=None, mask=None):
        # floor > 0 stops the near-zero rim of a lesion's spatial envelope
        # from being balanced up as if it were lesion core.
        nonzero_mask = target.abs() > self.floor if self.floor > 0 else target != 0
        if mask is None:
            n_nonzero = nonzero_mask.float().sum().clamp(min=1.0)
            n_zero = target.numel() - n_nonzero
        else:
            # The population the balance is computed OVER, not just the
            # contributions. Excluding voxels via `weight` alone leaves
            # n_zero counting them, so the inverse-frequency ratio stays
            # inflated and the nonzero class ends up overweighted --
            # measured at 58% of gradient mass instead of 40%.
            keep = mask.to(target.dtype)
            nonzero_mask = nonzero_mask & (keep > 0)
            n_nonzero = nonzero_mask.float().sum().clamp(min=1.0)
            n_zero = (keep.sum() - n_nonzero).clamp(min=0.0)
            if self.tissue_weight is not None:
                n_zero_full = target.numel() - n_nonzero
                w_legacy = torch.where(nonzero_mask, n_zero_full / n_nonzero, torch.ones_like(target))
                if weight is not None:
                    w_legacy = w_legacy * weight
                denom = w_legacy.sum().clamp(min=1e-8)              # the control's normaliser, air included
                tau = float(self.tissue_weight)
                w = torch.where(nonzero_mask, w_legacy, w_legacy * tau) * keep
                return (w * self._err(pred, target)).sum() / denom
        nonzero_weight = n_zero / n_nonzero
        # nonzero_weight is only ever applied where nonzero_mask is True, so
        # its value on an all-background batch (where n_nonzero was clamped
        # up from 0) is harmless: torch.where uses the ones branch
        # everywhere in that case regardless of what nonzero_weight computed to.
        w = torch.where(nonzero_mask, nonzero_weight, torch.ones_like(target))
        if mask is not None:
            w = w * keep
        if weight is not None:
            # A spatial multiplier on top of the class balance (e.g. the
            # ground-truth ROI); the normalisation below absorbs its scale.
            w = w * weight
        return (w * self._err(pred, target)).sum() / w.sum().clamp(min=1e-8)


class SoftBinaryDiceLoss(nn.Module):
    """Dice on one probability channel against a soft [0, 1] target.

    SoftDiceLoss takes integer class indices and softmaxes over classes, so
    it does not apply to a single sigmoid channel. The target is left soft
    rather than binarized: |reg_target| is already in [0, 1) and exactly
    zero where the generator did not touch, so no cutoff has to be picked.
    Dice rather than L1 because it scores overlap, which is what the
    thresholded map is finally judged on.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, prob, target):
        p = prob.reshape(prob.shape[0], -1)
        t = target.reshape(target.shape[0], -1)
        inter = (p * t).sum(dim=1)
        denom = p.sum(dim=1) + t.sum(dim=1)
        return (1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)).mean()


def _base_grid(shape, device, dtype):
    ax = [torch.linspace(-1, 1, s, device=device, dtype=dtype) for s in shape]
    gz, gy, gx = torch.meshgrid(*ax, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1)[None]


def warp(vol, flow):
    """Resample vol by flow, which is a displacement in normalized grid units.

    align_corners=False matches _inject_pathology's own grid_sample convention.
    """
    grid = _base_grid(vol.shape[2:], vol.device, flow.dtype) + flow.permute(0, 2, 3, 4, 1)
    return F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=False)


def label_dice(a, b, smooth=1.0):
    """Mean Dice over label channels. Differentiable: returning a float here
    detaches the graph and trains nothing."""
    inter = (a * b).sum(dim=(2, 3, 4))
    den = a.sum(dim=(2, 3, 4)) + b.sum(dim=(2, 3, 4))
    return ((2 * inter + smooth) / (den + smooth)).mean()


# --- 4. Initialization & Optimizations ---
device = target_device
if args.cudnn_benchmark:
    torch.backends.cudnn.benchmark = True   # OPT-IN (2026-10-05): with it on unconditionally, a from-scratch run died at step 0
    # with 'CUDA error: an illegal memory access' (speed_bench 10168248, base config); the forks without it run fine.
# BatchNorm is networks.py's default, matching SynthSeg's batch_norm=-1.
# regression_channels=1: a second head predicting a signed
# severity/direction map alongside segmentation. The head has no output
# activation -- the TARGET is tanh-bounded, not the head, so its output is
# unbounded in principle. See the regression
# target construction in the training loop for why tanh, and why the
# scale it is computed relative to. A third, probability head (P(normal
# tissue) at each voxel) was prototyped and reverted -- needs real design
# consideration before committing to it, not a decision to make alongside
# everything else changing here. networks.py's probability_channels
# parameter still exists (defaults to 0, i.e. omitted) for whenever that
# gets revisited.
if args.task == "register":
    # moving + fixed in, a 3-component displacement field out. No auxiliary
    # heads: the flow is the only output.
    model = SynthSegUNet(in_channels=2, out_channels=3,
                         attention=args.attention).to(device)
    _fwd = model
else:
    # kwargs only when set, so the default call is the one every old run made
    _head_kw = {}
    if args.reg_head != "linear":
        _head_kw["reg_head"] = args.reg_head
    if args.tissue_rel:
        _head_kw["tissue_rel"] = True
        if args.tissue_refs:
            _head_kw["tissue_refs"] = args.tissue_refs
        if args.tissue_vec != "none":
            _head_kw["tissue_vec"] = args.tissue_vec
        if args.tissue_pool != "own":
            _head_kw["tissue_pool"] = args.tissue_pool
        if args.tissue_soft:
            _head_kw["tissue_soft"] = True
        if args.tissue_unroll:
            _head_kw["tissue_unroll"] = args.tissue_unroll
        if (args.reg_hidden, args.reg_depth, args.reg_kernel) != (32, 2, 1):
            _head_kw.update(reg_hidden=args.reg_hidden, reg_depth=args.reg_depth, reg_kernel=args.reg_kernel)
    try:
        model = SynthSegUNet(out_channels=len(LABEL_LIST_SEGMENTATION), regression_channels=1,
                             probability_channels=0, attention=args.attention, attention_identity_init=args.attention_identity_init,
                             attention_dense=args.attention_dense, seg_prior=args.seg_prior, **_head_kw).to(device)
    except TypeError as e:
        if not _head_kw:
            raise
        raise SystemExit(f"networks.SynthSegUNet does not take {sorted(_head_kw)} (stale networks.py?): {e}")
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
        log_print("channels_last_3d: model weights converted; training input converted per step")
    _fwd = torch.compile(model) if args.compile else model   # forward handle only; saving/loading use `model`
    if args.compile: log_print("torch.compile: training forward compiled (first steps include compile time)")
    if args.bn_per_sample:
        from networks import per_sample_bn
        print(f"bn_per_sample on ({per_sample_bn(model)} layers)", flush=True)

if args.init_encoder:
    # Encoder only (enc1-4 + bottleneck); decoder and both heads stay random.
    _ck = torch.load(args.init_encoder, map_location=device)
    _sd = _ck["model_state_dict"] if isinstance(_ck, dict) and "model_state_dict" in _ck else _ck
    _keep = {k: v for k, v in _sd.items() if k.split(".")[0] in ("enc1", "enc2", "enc3", "enc4", "bottleneck")}
    _res = model.load_state_dict(_keep, strict=False)
    _unexpected = [k for k in _res.unexpected_keys]
    if _unexpected:
        raise RuntimeError(f"--init_encoder: unexpected keys {_unexpected[:5]}")
    log_print(f"Encoder initialised from {args.init_encoder}: {len(_keep)} tensors "
              f"({sum(v.numel() for v in _keep.values()) / 1e6:.1f}M params); "
              f"note: {_ck.get('note', '') if isinstance(_ck, dict) else ''}")

# Uncertainty weighting (Kendall et al., "Multi-task learning using
# uncertainty to weigh losses") instead of a fixed --regression_weight
# constant. Verified directly: holding loss_seg/loss_reg at their rough
# observed starting scales (4.0 / 0.09, a 44x raw ratio) and running
# gradient descent on log_sigma alone converges to precision_seg*loss_seg
# = 0.5 and precision_reg*loss_reg = 1.0 -- a fixed, known 2x ratio
# (a consequence of the Gaussian-style vs Laplace-style forms below, not
# an arbitrary residual), regardless of how far apart the raw losses
# started. Initialized to log(1)=0 -- equal weighting at step 0, same as
# regression_weight=1.0's old default -- then adapted by gradient descent
# from there based on each task's own observed scale, not a hand-picked
# number.
log_sigma_seg = torch.zeros(1, device=device, requires_grad=True)
log_sigma_reg = torch.zeros(1, device=device, requires_grad=True)

# Separate parameter group at a much larger lr than the network's 1e-4:
# verified directly that sharing the network's own lr takes ~20,000 steps
# to reach the predicted optimum (holding losses at their observed 4.5/0.1
# scale, simulated convergence matched the real run's logged sigma values
# almost to the decimal at steps 100 and 400) -- 4% of a 500,000-step run
# before the balance this exists to provide has any real effect yet.
#
# --sigma_lr controls this rate (default 1e-2, the value that simulation
# picked). That simulation held both task losses fixed to isolate the
# weighting mechanism in isolation -- it verified the mechanism converges
# fast, not that converging this fast is harmless to a network whose
# shared trunk both tasks depend on. A real run reached precision_reg~16x
# by step 500; if that swing turns out to visibly destabilize training
# (segmentation degrading, regression output going widespread/unlocalized
# shortly after), a smaller value here is the lever to soften it.
# --freeze_trunk: requires_grad is untouched by load_state_dict, so freezing here
# (before the optimizer) holds across --resume / --init_encoder.
if args.freeze_trunk:
    _trainable, _n_tr, _n_tot = freeze_trunk(model)
    log_print(f"freeze_trunk: {_n_tr} trainable of {_n_tot} parameters")
    if not (resume or args.init_encoder):
        log_print("WARNING: --freeze_trunk without --resume/--init_encoder: the frozen trunk is random")
else:
    _trainable = list(model.parameters())
if args.trunk_lr_mult != 1.0 and not args.freeze_trunk:
    from target_utils import is_reg_head_key
    _head = [p for n, p in model.named_parameters() if is_reg_head_key(n)]
    _trunk = [p for n, p in model.named_parameters() if not is_reg_head_key(n)]
    _param_groups = [{"params": _trunk, "lr": 1e-4 * args.trunk_lr_mult}, {"params": _head, "lr": 1e-4}]
    log_print(f"trunk_lr_mult {args.trunk_lr_mult}: trunk {sum(p.numel() for p in _trunk)} params at lr {1e-4 * args.trunk_lr_mult:.1e}, "
              f"head {sum(p.numel() for p in _head)} params at lr 1e-4")
else:
    _param_groups = [{"params": _trainable, "lr": 1e-4}]
if args.uncertainty_weighting:
    _param_groups.append({"params": [log_sigma_seg, log_sigma_reg], "lr": args.sigma_lr})
optimizer = torch.optim.Adam(_param_groups)
_base_lr = [g["lr"] for g in optimizer.param_groups]   # --lr_decay_from scales these per step
if args.grad_balance != "none":
    if args.uncertainty_weighting or args.freeze_trunk:
        raise SystemExit("--grad_balance excludes --uncertainty_weighting and --freeze_trunk")
    from target_utils import is_reg_head_key as _is_reg_key
    _gb_trunk = [p for n, p in model.named_parameters() if not _is_reg_key(n) and not n.startswith("final_conv")]
    _gb_last = [float("nan")] * 3
    _gb_seg_head = [p for n, p in model.named_parameters() if n.startswith("final_conv")]
    _gb_reg_head = [p for n, p in model.named_parameters() if _is_reg_key(n)]
    if args.grad_balance_every > 1 and args.grad_balance != "imtl":
        raise SystemExit("--grad_balance_every > 1 is implemented for imtl only (its weights sum to 1)")
    if args.grad_balance_every > 1:
        log_print(f"grad_balance_every {args.grad_balance_every}: balanced weights refreshed every {args.grad_balance_every} steps, "
                  "single backward in between (heads rescaled to their own task; trunk alpha at most that many steps old)")
    log_print(f"grad_balance {args.grad_balance}: {sum(p.numel() for p in _gb_trunk)} trunk parameters get the balanced "
              "seg/reg gradient every step; each head keeps its own loss")


def _gb_combine(gs, gr, rule):
    """Trunk gradient = ws*g_seg + wr*g_reg, written into p.grad. Returns (alpha_seg, |g_seg|, |g_reg|) in the input units.
    imtl: alpha with (alpha g1 + (1-alpha) g2) . u1 == (alpha g1 + (1-alpha) g2) . u2, u = g/|g|; closed form
    alpha = (|g2| - g1.g2/|g1|) / ((|g1| - g1.g2/|g2|) + (|g2| - g1.g2/|g1|)), always in [0, 1].
    norm: each task rescaled to the mean of the two norms, then averaged. Missing / zero gradient of one task -> the other alone."""
    pairs = [(p, a, b) for p, a, b in zip(_gb_trunk, gs, gr) if a is not None or b is not None]
    z = lambda t, ref: t if t is not None else torch.zeros_like(ref)
    ss = sum(((z(a, b)) ** 2).sum() for _, a, b in pairs); rr = sum(((z(b, a)) ** 2).sum() for _, a, b in pairs)
    sr = sum((z(a, b) * z(b, a)).sum() for _, a, b in pairs)
    ns, nr = ss.sqrt(), rr.sqrt()
    if nr <= 0 or ns <= 0:
        ws, wr = (1.0, 0.0) if nr <= 0 else (0.0, 1.0)
    elif rule == "imtl":
        num = nr - sr / ns; den = (ns - sr / nr) + num
        a = (num / den).clamp(0.0, 1.0) if float(den.abs()) > 1e-12 else torch.full_like(ns, 0.5)
        ws, wr = a, 1.0 - a
    else:
        m = 0.5 * (ns + nr); ws, wr = 0.5 * m / ns, 0.5 * m / nr
    for p, a, b in pairs:
        p.grad = ws * z(a, b) + wr * z(b, a)
    wsum = float(ws) + float(wr)
    return (float(ws) / wsum if wsum > 0 else float("nan")), float(ns), float(nr)
scaler = torch.amp.GradScaler('cuda', enabled=not args.bf16)   # --bf16: no scaling needed, get_scale() -> 1.0

criterion_ce = nn.CrossEntropyLoss()
criterion_ce_none = nn.CrossEntropyLoss(reduction='none')
# No lesion_idx/lesion_weight: the classification target is now always
# seg_clean_idx (see the target construction below), which never contains
# a Lesion class to weight -- lesion_idx=None here falls back to the plain
# unweighted foreground mean (see SoftDiceLoss.forward above).
criterion_dice = SoftDiceLoss()
# L1 rather than MSE: the target's scale varies a lot by how large a
# lesion's mu/sigma offset is (tens of raw GMM units, per the k_rose
# discussion), and MSE's squared penalty would let a handful of
# large-offset voxels dominate the gradient over the much more common
# small/no-difference ones. L1 weights all magnitudes more evenly.
# Weighted, not plain: see WeightedRegressionLoss's own docstring for why
# an unweighted version actively rewards collapsing to zero everywhere.
if args.reg_tissue_weight is not None and not args.reg_exclude_air:
    raise SystemExit("--reg_tissue_weight needs --reg_exclude_air")
criterion_regression = WeightedRegressionLoss(floor=args.target_floor, tissue_weight=args.reg_tissue_weight, quantile=args.reg_quantile)
if args.reg_quantile != 0.5:
    print(f"[loss] reg_quantile on (q={args.reg_quantile}): pinball regression toward the q-quantile of the deviation", flush=True)

# --- Fixed validation set: the epistemic-uncertainty check ------------------
# Every case in the training log's periodic dice line is freshly drawn --
# a checkpoint "getting worse" is indistinguishable from "drew a harder
# case this time." Evaluating the SAME frozen set every --validation_every
# steps removes that confound: if per-case scores on this fixed set stay
# flat while training continues, that's a real ceiling, not sampling noise.
# See pregenerate_validation_set.py for how this directory gets built.
_val_case_paths = None
if args.validation_dir:
    _val_case_paths = sorted(
        os.path.join(args.validation_dir, f) for f in os.listdir(args.validation_dir)
        if f.startswith("val_case_") and f.endswith(".pt")
    )
    log_print(f"Fixed validation set: {len(_val_case_paths)} cases from {args.validation_dir}, "
              f"evaluated every {args.validation_every} steps.")


def evaluate_validation_set(step):
    """Runs the frozen validation set through the CURRENT model, batching
    validation_batch_size cases per forward pass.

    The classification target is always seg_clean_idx for both streams, so
    there is no Lesion class in any target to score. Tracks mean_fg_dice
    (anatomical segmentation), the regression loss against a zero-output
    baseline, and reg_auc (separability of |pred|). Always the same frozen
    set, so per-checkpoint sampling noise is not a confound.
    """
    if not _val_case_paths:
        return
    t_val_start = time.time()
    was_training = model.training
    model.eval()

    all_fg_dice, all_reg_l1, all_zero_l1 = [], [], []
    all_reg_auc, all_reg_auc_vis, all_reg_bal = [], [], []
    bs = max(1, args.validation_batch_size)
    with torch.no_grad():
        for batch_start in range(0, len(_val_case_paths), bs):
            batch_paths = _val_case_paths[batch_start:batch_start + bs]
            records = [torch.load(p, map_location='cpu') for p in batch_paths]

            # Raw (pre-normalization) tensors kept around specifically for
            # the regression target -- same reasoning as the training loop:
            # the difference has to be computed before normalize_per_sample
            # divides each stream by its own max, or it partly measures the
            # normalization cascade instead of real pathology.
            clean_raw = torch.stack([
                r['image_clean'].to(device).float() for r in records]).unsqueeze(1)
            path_raw = torch.stack([
                r['image_path'].to(device).float() for r in records]).unsqueeze(1)
            # --norm_brain > 0: validation is deterministic, always the brain frame
            val_brain = args.norm_brain > 0
            if val_brain:
                log_print("validation: brain-frame percentiles (--norm_brain > 0)")
            clean_masks = ([brain_frame_mask(r['seg_clean_idx'].to(device)).reshape(clean_raw[0].shape) for r in records]
                           if val_brain else None)
            path_masks = ([brain_frame_mask(r['seg_path_idx'].to(device)).reshape(path_raw[0].shape) for r in records]
                          if val_brain else None)
            clean_imgs = normalize_per_sample(clean_raw, masks=clean_masks)
            path_imgs = normalize_per_sample(path_raw, masks=path_masks)
            # ALWAYS the healthy map for both streams -- matches the
            # training loop, never seg_path_idx.
            clean_tgts = torch.stack([r['seg_clean_idx'].to(device).long() for r in records])

            img = torch.cat([clean_imgs, path_imgs], dim=0)
            target = torch.cat([clean_tgts, clean_tgts], dim=0)
            # Same rescaling as the main training loop's reg_target_path --
            # see that construction for why raw units and dividing by
            # img_clean's own scale both fail at inference, and why tanh
            # rather than left unbounded. Looped per sample (this set is
            # small and only evaluated every --validation_every steps)
            # since each case's (p_lo, p_hi) is its own, not a
            # batch-pooled one.
            reg_target_path_list = []
            for j in range(path_raw.shape[0]):
                p_lo_j, p_hi_j = frame_percentiles(path_raw[j].clamp(min=0), None if path_masks is None else path_masks[j])
                scale_j = (p_hi_j - p_lo_j).clamp(min=1e-6)
                reg_target_path_list.append(torch.tanh((path_raw[j] - clean_raw[j]) / scale_j))
            reg_target_path = torch.stack(reg_target_path_list, dim=0)
            if (args.target_lesion_only or args.target_region > 0) and LESION_COMPACT_IDX is not None:
                _les = torch.stack([r['seg_path_idx'].to(device) == LESION_COMPACT_IDX
                                    for r in records]).unsqueeze(1)
                if args.target_lesion_only:
                    reg_target_path = reg_target_path * _les.to(reg_target_path.dtype)
                if args.target_region > 0:
                    reg_target_path = apply_target_region(reg_target_path, _les, path_raw - clean_raw, args.target_region)
            reg_target = torch.cat([torch.zeros_like(clean_raw), reg_target_path], dim=0)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16 if args.bf16 else torch.float16):
                preds, reg_out = model(img)

            probs = F.softmax(preds.float(), dim=1)
            oh = F.one_hot(target, probs.shape[1]).permute(0, 4, 1, 2, 3).float()
            p = probs.view(probs.size(0), probs.shape[1], -1)
            t = oh.view(oh.size(0), oh.shape[1], -1)
            inter = (p * t).sum(-1)
            den = p.sum(-1) + t.sum(-1)
            raw = 2. * inter / den.clamp(min=1e-6)   # (2*n_batch, n_classes)

            n_batch = len(records)
            present = t.sum(-1) > 0
            present[:, 0] = False   # exclude background from the foreground mean
            for j in range(raw.shape[0]):
                if present[j].any():
                    all_fg_dice.append(raw[j][present[j]].mean().item())

            # Regression error only on the PATHOLOGICAL half -- the clean
            # half's target is trivially zero everywhere, and including it
            # would dilute the error toward "how well does it predict zero"
            # rather than "how well does it predict actual pathology".
            reg_err = (reg_out.float()[n_batch:] - reg_target[n_batch:]).abs().mean(dim=(1, 2, 3, 4))
            all_reg_l1.extend(reg_err.tolist())

            # Same loss for a zero-output head. Balanced form, which is
            # what the loss reduces to once weight.sum() cancels the
            # inverse-frequency weight; plain L1 would flatter silence.
            _p = reg_out.float()[n_batch:]
            _t = reg_target[n_batch:]
            for _b in range(_p.shape[0]):
                _nz = _t[_b] != 0
                if _nz.sum() < 50 or (~_nz).sum() < 50:
                    continue
                _e = (_p[_b] - _t[_b]).abs()
                all_reg_bal.append(0.5 * (float(_e[_nz].mean()) + float(_e[~_nz].mean())))
                all_zero_l1.append(0.5 * float(_t[_b][_nz].abs().mean()))

            # Separability of |pred| between touched and untouched voxels:
            # Dice after thresholding follows this ranking, not the predicted
            # magnitude. Subsampled per case to bound cost.
            def _auc(score, positive):
                pos, neg = score[positive], score[~positive]
                if pos.numel() < 50 or neg.numel() < 50:
                    return None
                pos = pos[torch.randperm(pos.numel(), device=pos.device)[:20000]]
                neg = neg[torch.randperm(neg.numel(), device=neg.device)[:20000]]
                ranks = torch.argsort(torch.argsort(torch.cat([pos, neg]))).float() + 1.0
                n = pos.numel()
                return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * neg.numel()))

            for _b in range(n_batch, reg_out.shape[0]):
                _s = reg_out.float()[_b, 0].abs().flatten()
                _t = reg_target[_b, 0].abs().flatten() > 0
                _a = _auc(_s, _t)
                if _a is not None:
                    all_reg_auc.append(_a)
                # Same, with the near-zero rim of the target counted as
                # background: the number a visible-lesion mask would give.
                _a = _auc(_s, reg_target[_b, 0].abs().flatten() > 0.02)
                if _a is not None:
                    all_reg_auc_vis.append(_a)

            # This runs with the model's own weights/optimizer state already
            # resident (unlike a standalone eval script), so headroom is
            # tight by design -- explicitly drop the large per-batch
            # tensors and clear the allocator's cache rather than waiting
            # on Python's refcounting to get to it before the next
            # iteration allocates again.
            del preds, reg_out, probs, oh, p, t, inter, den, raw, present, reg_err
            del clean_raw, path_raw, clean_imgs, path_imgs, clean_tgts, img, target, reg_target, records
            torch.cuda.empty_cache()

    if was_training:
        model.train()
        if args.freeze_trunk:
            (frozen_bn_batch(model) if args.freeze_bn_batch else frozen_bn_eval(model))
    elapsed = time.time() - t_val_start
    if all_reg_l1:
        _reg_mean = sum(all_reg_l1) / len(all_reg_l1)
        _zero_mean = sum(all_zero_l1) / max(len(all_zero_l1), 1)
        # Below 100% the head beats predicting zero; at or above it does not.
        _bal_mean = sum(all_reg_bal) / max(len(all_reg_bal), 1)
        _vs_zero = 100.0 * _bal_mean / _zero_mean if _zero_mean > 0 else float("nan")
        _auc_str = (f"reg_auc: mean={sum(all_reg_auc)/len(all_reg_auc):.4f}  "
                    if all_reg_auc else "")
        _auc_str += (f"reg_auc_vis: mean={sum(all_reg_auc_vis)/len(all_reg_auc_vis):.4f}  "
                     if all_reg_auc_vis else "")
        log_print(f"[validation @ step {step}] n={len(_val_case_paths)}  "
                  f"regression_l1: mean={_reg_mean:.4f} "
                  f"median={sorted(all_reg_l1)[len(all_reg_l1)//2]:.4f}  "
                  f"balanced: mean={_bal_mean:.4f} vs zero-head {_zero_mean:.4f} "
                  f"({_vs_zero:.0f}% of silence)  "
                  + _auc_str
                  + f"mean_fg_dice: mean={sum(all_fg_dice)/max(len(all_fg_dice),1):.4f}  "
                  + f"elapsed={elapsed:.1f}s")
    else:
        log_print(f"[validation @ step {step}] no cases produced a result (unexpected) -- "
                  f"elapsed={elapsed:.1f}s")


def _rng_state_dict():
    # weights_only-safe: numpy's MT key as a list, not an ndarray
    st = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (st[0], st[1].tolist(), int(st[2]), int(st[3]), float(st[4])),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _rng_state_restore(st):
    py = st["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    n = st["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(st["torch"].cpu())
    n_saved = len(st["cuda"])
    n_here = torch.cuda.device_count() if torch.cuda.is_available() else 0
    for d in range(min(n_saved, n_here)):
        torch.cuda.set_rng_state(st["cuda"][d].cpu(), device=d)
    if n_saved != n_here:
        log_print(f"WARNING: CUDA RNG state saved for {n_saved} device(s), {n_here} visible; "
                  f"restored {min(n_saved, n_here)}, the rest keep the --seed stream")


def _load_weights(sd):
    if args.reg_head == "linear" and not args.tissue_rel and not args.attention:
        model.load_state_dict(sd)       # strict, as before
        return False
    # strict, except a regression head of another shape (old linear checkpoint,
    # new mlp / tissue_rel head): that one starts fresh
    if args.reinit_head:
        n_drop = sum(is_reg_head_key(k) for k in sd)
        sd = {k: v for k, v in sd.items() if not is_reg_head_key(k)}
        log_print(f"reinit_head: dropped {n_drop} regression-head tensors from the checkpoint; fresh head of the same shape")
    reinit, ck_head, my_head = load_weights_keep_head(model, sd)
    if reinit:
        log_print(f"regression head re-initialised (checkpoint head: {ck_head}, requested: {my_head})")
    return reinit


if resume:
    checkpoint = torch.load(resume_ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # New-style checkpoint with full training state
        _head_reinit = _load_weights(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and (args.freeze_trunk or _head_reinit):
            # the saved moments belong to another parameter set
            log_print("optimizer state not loaded ("
                      + ("--freeze_trunk" if args.freeze_trunk else "new regression head")
                      + "): Adam moments start fresh")
        elif "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError as e:
                # A checkpoint saved before log_sigma_seg/reg moved into
                # their own parameter group has one param group where this
                # optimizer now has two -- load_state_dict refuses a
                # mismatched group count outright rather than guessing
                # which old group maps to which new one. Falls back to
                # fresh Adam moment estimates for every parameter (a
                # temporary re-warm-up over the next couple hundred steps,
                # not a correctness issue) rather than crashing the
                # resume entirely -- model weights and log_sigma's own
                # values are unaffected, loaded separately below.
                log_print(f"Optimizer state didn't load ({e}) -- the parameter groups changed (a head-only "
                          "checkpoint resumed with the trunk unfrozen, --trunk_lr_mult, or a pre-log_sigma checkpoint). "
                          "Adam moments start fresh for all parameters; model weights are unaffected.")
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        # In-place (.copy_, not reassignment) so log_sigma_seg/log_sigma_reg
        # stay the SAME tensor objects the optimizer above was already
        # given -- reassigning the variable name would silently disconnect
        # them from optimizer.param_groups, and Adam would start optimizing
        # a stale, untracked copy instead. Falls back to the log(1)=0
        # (equal weighting) default for a checkpoint saved before
        # uncertainty weighting existed at all -- not a resume error, just
        # nothing to restore for a parameter that didn't exist yet.
        if "log_sigma_seg" in checkpoint and "log_sigma_reg" in checkpoint:
            with torch.no_grad():
                log_sigma_seg.copy_(checkpoint["log_sigma_seg"].to(device))
                log_sigma_reg.copy_(checkpoint["log_sigma_reg"].to(device))
            log_print(f"Restored learned task weighting: log_sigma_seg={log_sigma_seg.item():.4f} "
                      f"log_sigma_reg={log_sigma_reg.item():.4f}")
        else:
            log_print("Checkpoint predates uncertainty weighting -- log_sigma_seg/reg start at "
                      "equal weighting (0.0) rather than being restored.")
        # after model/optimizer: the checkpoint's stream wins over the startup --seed
        if "rng_state" in checkpoint and not args.no_rng_restore:
            _rng_state_restore(checkpoint["rng_state"])
            log_print("RNG state restored from checkpoint"
                      + (f" (overrides --seed {args.seed})" if args.seed is not None else ""))
        elif "rng_state" in checkpoint:
            log_print(f"--no_rng_restore: checkpoint RNG state ignored; continuing from --seed {args.seed}")
        else:
            log_print(f"checkpoint has no RNG state; continuing from --seed {args.seed}")
        log_print(f"Loaded full training state from {resume_ckpt_path}.")
    else:
        # Legacy checkpoint: a raw model state_dict only (no optimizer/scaler state)
        _load_weights(checkpoint)
        log_print(
            f"Loaded legacy checkpoint (model weights only) from {resume_ckpt_path}. "
            "Optimizer and GradScaler state were not saved in this checkpoint, so they start fresh."
        )
        log_print(f"checkpoint has no RNG state; continuing from --seed {args.seed}")

_ema = None
if args.ema > 0:
    _ema = {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}
    if resume and isinstance(checkpoint, dict) and "ema_state_dict" in checkpoint:
        for k, v in checkpoint["ema_state_dict"].items():
            if k in _ema:
                _ema[k].copy_(v.to(_ema[k].device))
        log_print(f"ema {args.ema}: ema_state_dict restored from the checkpoint ({len(_ema)} tensors)")
    else:
        log_print(f"ema {args.ema}: initialised from the current weights ({len(_ema)} tensors)")
if args.lr_decay_from > 0:
    log_print(f"lr_decay_from {args.lr_decay_from}: cosine to {args.lr_final_mult} x base lr at step {target_steps}")

if args.prefetch and resume:
    gen = Prefetcher(gen, depth=2)
    log_print("Prefetch enabled (background generation, depth 2).")

# --- 5. Optimized Training Loop ---
log_print(f"Starting {RUN_NAME}. Target: {target_steps} steps.")

if start_step >= target_steps:
    log_print(f"Checkpoint step {start_step} already meets/exceeds target_steps={target_steps}. Nothing to do.")

# Brain of a compact-index label map, mirroring synthstrip -b 0: everything except
# background, sulcal CSF (16 -> 4), optic chiasm (32 -> 6) and the extracerebral
# classes 34-46 (compact 8-19). Used by --norm_brain.
_BRAIN_LUT = torch.zeros(len(LABEL_LIST_SEGMENTATION), dtype=torch.bool)
_BRAIN_LUT[[1, 2, 3, 5, 7]] = True
_BRAIN_LUT[20:] = True


def brain_frame_mask(seg_idx):
    return _BRAIN_LUT.to(seg_idx.device)[seg_idx.long()]


def frame_percentiles(xi, brain=None):
    """[p1, p99] of one volume over xi > 0 (legacy frame) or, with a brain mask, over
    the brain voxels > 0 -- the same rule compare_datasets.normalize_like_training
    applies under --norm_mask."""
    fg = xi[(xi > 0) & brain.reshape(xi.shape)] if brain is not None else xi[xi > 0]
    if fg.numel() > 0:
        p_lo, p_hi = torch.quantile(fg, torch.tensor([0.01, 0.99], device=xi.device, dtype=fg.dtype))
    else:
        p_lo, p_hi = torch.tensor(0.0, device=xi.device), torch.tensor(1.0, device=xi.device)
    return p_lo, p_hi


def normalize_per_sample(x, precomputed=None, masks=None):
    """Scale each sample to [0, 1] via SynthSeg's own inference-time
    normalization: clip to [1st, 99th] percentile, then rescale that
    clipped range to [0, 1].

    This replaced a plain min/max version. Plain max has no robustness at
    all -- a single hot voxel (or, during training, a bright lesion) sets
    the divisor directly, verified directly to distort the shared-scale
    computation by 3x for a lesion occupying just 0.1% of a volume, and to
    do so MORE at training time than compare_datasets.py's inference-time
    normalization did, since that already used a percentile clip on the
    upper end. That gap is exactly what this closes: SAME 1st/99th
    percentile formula, used identically at train and inference, not
    approximated one place and exact the other.

    Per-sample (looped over the batch), not pooled: a step's batch can hold
    two volumes (clean + pathological) with independent intensity scales,
    and pooling would let one rescale the other. Looped rather than a
    single vectorized quantile call because the number of foreground
    voxels varies per sample, which torch.quantile can't reduce over
    per-batch-element in one call.

    precomputed: optional list of (p_lo, p_hi) tuples, one per batch
    element, for a caller that already computed the same volume's
    percentiles for another purpose -- the regression target's own rescale
    factor is the same quantile call on the same tensor, and computing it
    twice was pure waste. None (default) computes internally exactly as
    before, so every other call site is unaffected.

    masks: optional list of per-sample brain masks (--norm_brain); the
    percentiles are then taken inside the mask (frame_percentiles).
    """
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        xi = x[i]
        if precomputed is not None:
            p_lo, p_hi = precomputed[i]
        else:
            p_lo, p_hi = frame_percentiles(xi, None if masks is None else masks[i])
        xi = xi.clamp(min=p_lo, max=p_hi) - p_lo
        scale = (p_hi - p_lo).clamp(min=1e-7)
        out[i] = (xi / scale).clamp(0, 1)
    return out

for i in range(start_step + 1, target_steps + 1):
    # Update the shared state so the curriculum sampler knows the current step
    train_state['step'] = i
    
    # 1. Fetch data FIRST -- paired_pathology_generator always yields both
    #
    # CUDA is asynchronous: without synchronize() these timers measure when
    # work was SUBMITTED, not when it finished. Since the generator also runs
    # on GPU, its queued work would otherwise be charged to whatever later op
    # happens to synchronize -- making gen look small and net look large.
    # Costs a small stall per step; remove once the numbers are understood.
    if _timing:
        torch.cuda.synchronize()
    t0 = time.time()
    case = next(gen)
    if _timing:
        torch.cuda.synchronize()
    t_gen = time.time() - t0
    t0 = time.time()
    
    # 2. Force gradients back ON for network pass
    torch.set_grad_enabled(True)

    if args.task == "register":
        # Second, independent draw: the fixed image is a DIFFERENT subject, so
        # no change to the generator's own per-case loading is needed.
        case_fixed = next(gen)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        mov_img = normalize_per_sample(
            case['image_path'].detach().to(device).float()[None, None])
        fix_img = normalize_per_sample(
            case_fixed['image_clean'].detach().to(device).float()[None, None])
        mov_lab = case['seg_clean_idx'].detach().to(device).long()[None]
        fix_lab = case_fixed['seg_clean_idx'].detach().to(device).long()[None]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16 if args.bf16 else torch.float16):
            flow = model(torch.cat([mov_img, fix_img], dim=1)) * args.flow_scale
        flow = flow.float()
        # One-hot at 56 channels on the full grid is the memory driver here;
        # built outside autocast so the Dice sums stay in float32.
        mov_oh = F.one_hot(mov_lab, 56).permute(0, 4, 1, 2, 3).float()
        fix_oh = F.one_hot(fix_lab, 56).permute(0, 4, 1, 2, 3).float()
        dice_term = 1.0 - label_dice(warp(mov_oh, flow), fix_oh)
        smooth_term = sum((flow.diff(dim=d) ** 2).mean() for d in (2, 3, 4))
        loss = dice_term + args.smoothness_weight * smooth_term

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if i % 100 == 0:
            with torch.no_grad():
                d_before = float(label_dice(mov_oh, fix_oh))
                d_after = 1.0 - float(dice_term)
            gen_label = "wait" if args.prefetch else "gen"
            log_print(f"Step {i:05d} | Loss: {float(loss):.4f} "
                      f"(dice {d_after:.4f} from {d_before:.4f} + smooth "
                      f"{float(smooth_term):.5f}) | {gen_label} {t_gen:.2f}s "
                      f"| net {time.time() - t0:.2f}s")
        if i % 5000 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"synthseg_{i:05d}.pth")
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "task": args.task, "step": i,
                        "rng_state": _rng_state_dict()}, ckpt_path)
            log_print(f"Saved Checkpoint: {ckpt_path}")
        del mov_oh, fix_oh, flow, mov_img, fix_img, case, case_fixed
        continue

    
    model.train()
    if args.freeze_trunk:
        (frozen_bn_batch(model) if args.freeze_bn_batch else frozen_bn_eval(model))   # trunk BN stays on its running statistics
    optimizer.zero_grad(set_to_none=True)
    
    # Raw generated image with positive intensities
    img_clean = case['image_clean'].detach().to(device, non_blocking=True).float().unsqueeze(0).unsqueeze(0)
    target_clean = case['seg_clean_idx'].detach().to(device, non_blocking=True).long().unsqueeze(0)
    # --norm_brain: one coin per anatomy; the clean row, the pathological row, its
    # target rescale and the viz all share it. P=0 draws nothing.
    brain_frame = args.norm_brain > 0 and np.random.rand() < args.norm_brain
    brain_clean = brain_frame_mask(target_clean).reshape(img_clean.shape) if brain_frame else None
    
    # Keep a reference to the uncorrupted image to visualize later
    clean_img_raw = img_clean.clone() 
    
    # Apply motion probabilistically (clean stream only -- see note below)
    motion_fired = do_motion and (np.random.rand() < args.motion_prob)
    if motion_fired:
        img_3d = img_clean.squeeze()
        img_3d = simulate_kspace_motion(img_3d, device, num_segments=np.random.randint(4, 12))
        img_clean = img_3d.unsqueeze(0).unsqueeze(0)

    # The pathological case is ALWAYS available (paired_pathology_generator
    # generates it unconditionally, sharing all downstream augmentation
    # randomness with the clean case). --do_pathology/--pathology_prob only
    # control whether THIS step's loss actually trains on it, matching the
    # existing --do_motion/--motion_prob pattern. NOTE: motion corruption is
    # only applied to the clean stream above -- combining it with the
    # pathological stream too is a reasonable extension but not done here to
    # avoid entangling two independent augmentations without being asked.
    use_pathology = do_pathology and (np.random.rand() < args.pathology_prob)
    if use_pathology:
        img_path = case['image_path'].detach().to(device, non_blocking=True).float().unsqueeze(0).unsqueeze(0)
        # ALWAYS the healthy map, never seg_path_idx -- the network is never
        # trained to produce a pathological segmentation, on the clean OR
        # the pathological stream. Any visualization of "where is the
        # pathology" is a post-hoc threshold of the regression head's
        # output layered on top of this healthy prediction, not a separate
        # trained class.
        #
        # seg_clean_idx is the UNDISPLACED map unless --displace_clean, in
        # which case the clean stream shares the pathological stream's
        # mass-effect/atrophy displacement and this target is aligned with
        # both images.
        target_path = case['seg_clean_idx'].detach().to(device, non_blocking=True).long().unsqueeze(0)


        # Regression target: (img_path - img_clean) / hi, where hi is the
        # SAME 99.5th-percentile robust scale compare_datasets.py's
        # normalize_like_training already computes for input normalization
        # -- NOT raw GMM units, and NOT divided by img_clean's own scale.
        #
        # Raw units failed at inference for a structural reason, not just
        # a scale-tuning one: two cases with identical RELATIVE contrast
        # but different absolute GMM draws produce IDENTICAL normalized
        # network inputs (normalize_per_sample removes absolute scale) but
        # DIFFERENT raw targets (verified directly -- 30 vs 15 units for a
        # 2x difference in absolute scale alone). The network's input
        # cannot distinguish these cases, yet a raw target demands two
        # different correct answers from the same input.
        #
        # Dividing by img_clean's own scale (an earlier version of this
        # fix) resolves that but creates a worse problem: img_clean does
        # not exist at inference on real data -- there is no healthy
        # counterfactual for a real scan. hi is computed from img_path
        # alone, so it is available identically whether img_path is a
        # synthetic GMM draw or a real clinical scan.
        #
        # Percentile, not max: max is exactly the cascade failure mode
        # this was designed to avoid (a bright lesion inflating its own
        # scale). Verified directly: for lesions under ~0.5% of volume
        # (most localized pathology), a 99.5th-percentile-style clip is
        # essentially untouched by the lesion (ratio to the lesion-free
        # case 1.001-1.003). It degrades gracefully rather than breaking
        # cleanly for large/diffuse pathology (Global Pathology, disease
        # exceeding ~1% of volume), where the percentile itself starts
        # drifting upward with lesion size -- a smaller, residual version
        # of the same cascade problem, not eliminated entirely.
        #
        # Scale is now the SAME (p99 - p1) range normalize_per_sample uses
        # for input normalization, not an independent percentile choice --
        # this expresses the target in exactly the units the network's own
        # input is normalized into, rather than a second, only loosely
        # related scale.
        #
        # tanh, not left unbounded: bounds the TARGET -- the head itself
        # has no output activation -- so the loss is computed in a space
        # the network can comfortably reach. Still signed (odd
        # function preserves brighter/darker direction) and still
        # exact-zero for untouched voxels under the generator's
        # shared-noise design -- tanh(0) = 0 exactly.
        brain_path = (brain_frame_mask(case['seg_path_idx'].to(device, non_blocking=True)).reshape(img_path.shape)
                      if brain_frame else None)
        p_lo, p_hi = frame_percentiles(img_path.clamp(min=0), brain_path)
        scale_path = (p_hi - p_lo).clamp(min=1e-6)
        reg_target_path = torch.tanh((img_path - img_clean) / scale_path)
        if (args.target_lesion_only or args.target_region > 0) and LESION_COMPACT_IDX is not None:
            _les = (case['seg_path_idx'].to(device, non_blocking=True) == LESION_COMPACT_IDX)
            if args.target_lesion_only:
                # The measured target has a halo of intensity change several
                # voxels beyond the lesion label; the real-data ground truth
                # does not. Zero the target outside the label.
                reg_target_path = reg_target_path * _les.reshape(reg_target_path.shape).to(reg_target_path.dtype)
            if args.target_region > 0:
                # sub-tau interior -> sign x C; rim, halo and air untouched
                reg_target_path = apply_target_region(reg_target_path, _les, img_path - img_clean, args.target_region)

    # 3. Robust Per-Volume Intensity Standardization (Zero Mean, Unit Variance)
    # Done ONCE at the end so physics operations aren't distorted by negative values.
    # Per-sample now (see normalize_per_sample) since the batch may contain
    # two differently-scaled volumes.
    img_clean_norm = normalize_per_sample(img_clean, masks=[brain_clean] if brain_frame else None)
    clean_img_vis = normalize_per_sample(clean_img_raw, masks=[brain_clean] if brain_frame else None)
    # The clean stream's own correct regression target is exactly zero
    # everywhere -- "no pathology present" is itself a real, informative
    # training example, not a case to skip.
    reg_target_clean = torch.zeros_like(img_clean)

    if use_pathology:
        # Cheap (two .amax() calls), so it runs every step rather than just
        # on viz steps -- only prints when the ratio is actually suspicious.
        diag.check_normalization_cascade(img_clean, img_path, step=i, log_print=log_print)
        # p_lo/p_hi already computed above for the regression target's own
        # rescale factor -- same tensor, same percentiles, reused here
        # rather than run through torch.quantile on ~4M voxels a second
        # time for the same answer.
        img_path_norm = normalize_per_sample(img_path, precomputed=[(p_lo, p_hi)])
        img = torch.cat([img_clean_norm, img_path_norm], dim=0)     # (2,1,D,H,W)
        target = torch.cat([target_clean, target_path], dim=0)      # (2,D,H,W)
        reg_target = torch.cat([reg_target_clean, reg_target_path], dim=0)  # (2,1,D,H,W)
    else:
        img = img_clean_norm
        target = target_clean
        reg_target = reg_target_clean

    # Motion consistency: ask the encoder to produce the same bottleneck
    # representation for the corrupted volume as for its motion-free
    # counterpart. The clean volume is appended to the SAME batch rather than
    # run as a second forward pass, so BatchNorm computes its statistics over
    # both and updates its running estimates once -- two separate passes in
    # train mode would update them twice per step with correlated data.
    use_consistency = motion_fired and args.motion_consistency
    n_corrupt = img.shape[0]
    if use_consistency:
        img = torch.cat([img, clean_img_vis], dim=0)
        target = torch.cat([target, target_clean], dim=0)
        # clean_img_vis is another motion-free clean copy -- same "no
        # pathology" target as reg_target_clean.
        reg_target = torch.cat([reg_target, reg_target_clean], dim=0)

    if args.channels_last:
        img = img.contiguous(memory_format=torch.channels_last_3d)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16 if args.bf16 else torch.float16):
        if use_consistency:
            preds, reg_out, feats = _fwd(img, return_features=True)
        else:
            preds, reg_out = _fwd(img)
        # fp32 for the seg loss: the Dice backward (softmax + arithmetic)
        # overflows fp16 under the 65536 loss scale, which cascades to nan
        # under --tissue_sub. Verified: fp16 nan'd by step ~385, fp32 stable.
        preds = preds.float()
        # Kept as its own variable (not folded straight into loss) so the
        # per-step log below can report it separately from the regression
        # term -- otherwise a single combined number can't tell you which
        # component is actually driving a change in total loss.
        # Row 1 is the pathological stream; its target is the HEALTHY map, and
        # the ROI is every healthy label the lesion overlaps. Shared by the
        # seg-CE weight and the regression weight below.
        _roi_mask = None
        if use_pathology and (args.roi_weight > 0 or args.reg_roi_weight > 0):
            _sc = target[1]
            _sp = case['seg_path_idx'].to(device, non_blocking=True).long()
            _roi_labels = torch.unique(_sc[_sp != _sc])
            _roi_labels = _roi_labels[_roi_labels != 0]
            _roi_mask = torch.isin(_sc, _roi_labels).float()
        # Which voxels the seg head is trained on. _segkeep is None
        # (all) or a (N,D,H,W) 0/1 mask shared by CE and Dice.
        _segkeep = None
        if use_pathology and args.seg_mask_lesion:
            _segkeep = torch.ones_like(target, dtype=torch.float32)
            _segkeep[1] = (case['seg_path_idx'].to(device, non_blocking=True) != LESION_COMPACT_IDX).float()
        if use_pathology and args.seg_clean_only:
            # row 0 only, sliced rather than masked: a masked-out row would still
            # contribute Dice's smooth term and halve the loss scale
            loss_ce = criterion_ce(preds[:1], target[:1])
            _segkeep = "clean_only"
        elif (args.roi_weight > 0 and _roi_mask is not None) or _segkeep is not None:
            # Weight is 1 everywhere else, so 0 reproduces the plain mean exactly.
            _w = torch.ones_like(target, dtype=torch.float32)
            if args.roi_weight > 0 and _roi_mask is not None:
                _w[1] = 1.0 + args.roi_weight * _roi_mask
            if _segkeep is not None:
                _w = _w * _segkeep
            loss_ce = (criterion_ce_none(preds, target) * _w).sum() / _w.sum().clamp(min=1.0)
        else:
            loss_ce = criterion_ce(preds, target)
        _rw = None
        if args.reg_roi_weight > 0 and _roi_mask is not None:
            _rw = torch.ones_like(reg_target)
            _rw[1, 0] = 1.0 + args.reg_roi_weight * _roi_mask
        if args.halo_weight > 0 and use_pathology and LESION_COMPACT_IDX is not None:
            # Hard negatives: zero-target voxels within HALO_RADIUS of the
            # lesion, where the image differs (PSF halo) but the target does
            # not. A balanced loss spreads the zero pool over the whole
            # volume, so these get almost no gradient without this.
            _les = (case['seg_path_idx'].to(device, non_blocking=True) == LESION_COMPACT_IDX).float()
            _halo = F.max_pool3d(_les[None, None], 2 * HALO_RADIUS + 1, stride=1, padding=HALO_RADIUS)[0, 0] * (1.0 - _les)
            if _rw is None:
                _rw = torch.ones_like(reg_target)
            _rw[1, 0] = _rw[1, 0] + args.halo_weight * _halo
        # Background is seg label 0. Shaped like reg_target so the loss can use
        # it as both the contribution mask and the balance population.
        _fg = (target != 0).unsqueeze(1).float() if args.reg_exclude_air else None
        if _segkeep == "clean_only":
            loss_seg = loss_ce + criterion_dice(preds[:1], target[:1])
        else:
            loss_seg = loss_ce + criterion_dice(preds, target, mask=_segkeep)
        # .float(): reg_out comes out of autocast as float16; L1 over ~4M
        # voxels in half precision loses precision in exactly the
        # small-magnitude regime (most of the volume, where the target is
        # near/exactly zero) that matters for this loss.
        if args.clean_weight > 0:
            # Split by stream: the balanced loss on rows that carry pathology,
            # a plain L1 toward zero on rows that carry none. A clean row is
            # one whose target is zero everywhere.
            _is_clean = reg_target.flatten(1).abs().amax(1) == 0
            loss_regression = reg_out.new_zeros(())
            if (~_is_clean).any():
                loss_regression = criterion_regression(
                    reg_out[~_is_clean].float(), reg_target[~_is_clean],
                    weight=None if _rw is None else _rw[~_is_clean],
                    mask=None if _fg is None else _fg[~_is_clean])
            if _is_clean.any():
                _c = reg_out[_is_clean].float().abs()
                if _fg is None:
                    _c = _c.mean()
                else:   # same exclusion, or air dominates this term instead
                    _m = _fg[_is_clean]
                    _c = (_c * _m).sum() / _m.sum().clamp(min=1e-8)
                loss_regression = loss_regression + args.clean_weight * _c
        else:
            loss_regression = criterion_regression(reg_out.float(), reg_target,
                                                   weight=_rw, mask=_fg)
        if args.uncertainty_weighting:
            # precision_seg uses the Gaussian/squared form, matching CE's
            # likelihood; precision_reg the Laplace/linear form, matching L1's.
            precision_seg = torch.exp(-2 * log_sigma_seg)
            precision_reg = torch.exp(-log_sigma_reg)
            loss = (precision_seg * loss_seg + log_sigma_seg
                   + precision_reg * loss_regression + log_sigma_reg).squeeze()
        else:
            loss = args.seg_weight * loss_seg + args.regression_weight * loss_regression

        if use_consistency:
            f_corrupt = feats[:n_corrupt]
            # detach: the clean encoding is the TARGET. Without this the loss
            # could be minimised by dragging the clean representation toward
            # the corrupted one, which is the opposite of the intent.
            f_clean = feats[n_corrupt:].detach()
            # Cosine rather than MSE: bottleneck activations have large and
            # drifting magnitudes (running_var there is ~1e3), so a raw MSE
            # would swamp the segmentation loss and its weight would need
            # retuning as training progresses. Cosine is scale-free and
            # bounded, and "match the direction of the encoding" is the
            # actual intent.
            cos = F.cosine_similarity(f_corrupt.flatten(2), f_clean.flatten(2), dim=1)
            loss_consistency = (1.0 - cos).mean()
            loss = loss + args.motion_consistency_weight * loss_consistency
    
    if not bool(torch.isfinite(loss)):
        # 2026-10-26: skip (do not backward) a non-finite step and NAME the draw. Under fp16 the GradScaler silently skipped these
        # and halved its scale each time (the fpi arm: 35 halvings, every BatchNorm running stat NaN, seg gradient underflowed to 0
        # from step 12.7k); under --bf16 there is no scaler, so one NaN loss would write NaN into the weights.
        globals()["_nonfinite_n"] = globals().get("_nonfinite_n", 0) + 1
        _m = case.get("metadata", {}) if isinstance(case, dict) and isinstance(case.get("metadata"), dict) else {}
        log_print(f"Step {i:05d} | NON-FINITE loss {float(loss):.4g} (seg {float(loss_seg):.4g} reg {float(loss_regression):.4g}) -- step SKIPPED "
                  f"({globals()['_nonfinite_n']} so far); draw " + str({k: _m.get(k) for k in ("mode_name", "donor", "self_patch", "alpha_texture", "alpha_bias", "center", "n_label") if k in _m})[:320]
                  + " " + str({k: case[k] for k in ("stem", "subject", "name", "path") if isinstance(case, dict) and k in case})[:160])
        # 2026-10-27: a skipped step never calls backward, so the whole autograd graph stays alive through `loss`/`preds`/`reg_out`
        # until the next iteration rebinds them -- the next forward then holds TWO graphs and OOMs (both fpi arms died at the step
        # right after their first skip: 35.6 GB allocated, 3.4 GB more requested). Drop the references and return the blocks now.
        loss = loss_seg = loss_regression = loss_ce = preds = reg_out = None
        optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        continue
    _gb_amortized = (args.grad_balance == "imtl" and args.grad_balance_every > 1 and i % args.grad_balance_every != 0
                     and _gb_last[0] == _gb_last[0] and 0.05 <= _gb_last[0] <= 0.95 and bool(loss_regression.requires_grad))
    if _gb_amortized:
        # one backward on the weighted sum with the LAST balanced weights: the trunk then holds ws*g_seg + wr*g_reg
        # exactly as the two-pass path would with those weights, and each head's gradient is divided by its own weight
        # so it keeps its own task's gradient (final_conv only ever sees the seg loss, the reg head only the reg loss)
        _ws = float(_gb_last[0]); _wr = 1.0 - _ws
        _seg_total = args.seg_weight * loss_seg + (args.motion_consistency_weight * loss_consistency if use_consistency else 0.0)
        scaler.scale(_ws * _seg_total + _wr * args.regression_weight * loss_regression).backward()
        for p in _gb_seg_head:
            if p.grad is not None: p.grad.div_(_ws)
        for p in _gb_reg_head:
            if p.grad is not None: p.grad.div_(_wr)
    elif args.grad_balance != "none":
        # two backward passes in the same AMP scale: seg (+ consistency) first, its trunk gradient set aside, then reg;
        # the trunk gets the balanced combination, each head only its own task's gradient; scaler.unscale_ below divides
        # the common scale out as usual
        _seg_total = args.seg_weight * loss_seg + (args.motion_consistency_weight * loss_consistency if use_consistency else 0.0)
        scaler.scale(_seg_total).backward(retain_graph=bool(loss_regression.requires_grad))
        _gs = [p.grad.detach().clone() if p.grad is not None else None for p in _gb_trunk]
        for p in _gb_trunk:
            p.grad = None
        if loss_regression.requires_grad:
            scaler.scale(args.regression_weight * loss_regression).backward()
        _gr = [p.grad for p in _gb_trunk]
        _gb_last = list(_gb_combine(_gs, _gr, args.grad_balance))
        _gb_last[1] /= scaler.get_scale(); _gb_last[2] /= scaler.get_scale()
    else:
        scaler.scale(loss).backward()
    
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if args.lr_decay_from > 0 and i >= args.lr_decay_from:
        _frac = min(1.0, (i - args.lr_decay_from) / max(1, target_steps - args.lr_decay_from))
        _m = args.lr_final_mult + (1.0 - args.lr_final_mult) * 0.5 * (1.0 + math.cos(math.pi * _frac))
        for _g, _b in zip(optimizer.param_groups, _base_lr):
            _g["lr"] = _b * _m
    scaler.step(optimizer)
    scaler.update()
    if _ema is not None:
        with torch.no_grad():
            for _k, _v in model.state_dict().items():
                if _k in _ema:
                    _ema[_k].mul_(args.ema).add_(_v.detach(), alpha=1.0 - args.ema)
    
    if i % 100 == 0:
        if _timing:
            torch.cuda.synchronize()
            # With prefetching, t_gen is queue-WAIT, not generation cost: near 0
        # means the producer is keeping up and generation is fully hidden;
        # consistently above 0 means generation is now the bottleneck again.
        gen_label = "wait" if args.prefetch else "gen"
        if args.uncertainty_weighting:
            _terms = (f"(seg {(precision_seg * loss_seg).item():.4f} "
                      f"+ reg {(precision_reg * loss_regression).item():.4f}) "
                      f"| sigma_seg {torch.exp(log_sigma_seg).item():.4f} "
                      f"sigma_reg {torch.exp(log_sigma_reg).item():.4f} ")
        else:
            _terms = (f"(seg {loss_seg.item():.4f} "
                      f"+ reg {(args.regression_weight * loss_regression).item():.4f}) ")
            if args.grad_balance != "none":
                _terms += f"| gb a_seg {_gb_last[0]:.2f} |g|seg {_gb_last[1]:.2e} reg {_gb_last[2]:.2e} "
        log_print(f"Step {i:05d} | Loss: {loss.item():.4f} " + _terms
                  + f"| {gen_label} {t_gen:.2f}s | net {time.time() - t0:.2f}s"
                  + (" | +pathology" if use_pathology else "")
                  + (f" | lr {optimizer.param_groups[0]['lr']:.2e}" if args.lr_decay_from > 0 else ""))

    if i % 500 == 0 and use_pathology:
        with torch.no_grad():
            # Batch element 1 is the pathological volume when it is present;
            # element 0 is always the clean stream, whose regression target
            # is trivially zero everywhere -- averaging it in would measure
            # "how well does it predict zero" rather than "how well does it
            # track real pathology".
            b = 1

            # .float() before reducing -- preds/reg_out come out of autocast
            # as float16, and summing ~4M voxels in half precision loses
            # accuracy in exactly the small-magnitude regime that matters.
            reg_l1_path = (reg_out.float()[b] - reg_target[b]).abs().mean().item()

            probs = F.softmax(preds.float(), dim=1)
            oh = F.one_hot(target, probs.shape[1]).permute(0, 4, 1, 2, 3).float()
            p = probs.view(probs.size(0), probs.shape[1], -1)
            t = oh.view(oh.size(0), oh.shape[1], -1)
            inter = (p * t).sum(-1)
            den = p.sum(-1) + t.sum(-1)
            # Raw Dice, no smoothing. The training loss uses smooth=1.0,
            # which is right for gradients but inflates the logged value.
            raw = (2. * inter / den.clamp(min=1e-6))[b]
            present = t[b].sum(-1) > 0
            present[0] = False                      # exclude background
            mean_fg = raw[present].mean().item() if present.any() else float('nan')

            # RUNNING_DIAG's methods are named for dice (record_dice,
            # rolling_dice_mean) from before the regression-map refactor --
            # reused here for the same rolling-mean mechanism, now tracking
            # regression L1 on the pathological stream instead. Renaming
            # them is a live_diagnostics.py change, not done here.
            RUNNING_DIAG.record_dice(reg_l1_path)
            log_print(f"Step {i} | path regression L1 {reg_l1_path:.4f} "
                      f"| mean fg dice {mean_fg:.4f} "
                      f"| L1_roll20 {RUNNING_DIAG.rolling_dice_mean(20):.4f}")

    if _val_case_paths and i % args.validation_every == 0:
        evaluate_validation_set(i)

    # Visual Check & Save to 'progress' folder
    if i % 1000 == 0:
        model.eval()
        with torch.no_grad():
            pred_idx = torch.argmax(preds, dim=1).cpu()   # (B,D,H,W)
            target_cpu = target.cpu()
            clean_cpu = clean_img_vis[0, 0].cpu()

            if not do_pathology:
                slice_idx = 80
                panels = [(clean_cpu[:, :, slice_idx], 'gray', f"Clean Input (Step {i})", None, None)]
                if do_motion:
                    panels.append((img.cpu()[0, 0][:, :, slice_idx], 'gray', "Network Input (Motion)", None, None))
                panels.append((target_cpu[0][:, :, slice_idx], FS_CMAP, "Ground Truth", 0, FS_N_LABELS - 1))
                panels.append((pred_idx[0][:, :, slice_idx], FS_CMAP, "Prediction", 0, FS_N_LABELS - 1))

                fig, ax = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
                if len(panels) == 1:
                    ax = [ax]
                for a, (data, cmap, title, vmin, vmax) in zip(ax, panels):
                    a.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                    a.set_title(title)
                    a.axis('off')
            else:
                # paired_pathology_generator always produces this case's
                # pathological version regardless of whether THIS step's
                # random draw (--pathology_prob) happened to train on it --
                # so always visualize it, recomputed fresh for a simple,
                # single code path (the extra forward pass only happens on
                # these 1-in-1000 visualization steps, cost is trivial).
                img_path_raw_vis = case['image_path'].detach().to(device, non_blocking=True).float().unsqueeze(0).unsqueeze(0)
                brain_path_vis = (brain_frame_mask(case['seg_path_idx'].to(device, non_blocking=True)).reshape(img_path_raw_vis.shape)
                                  if brain_frame else None)
                img_path_vis = normalize_per_sample(img_path_raw_vis, masks=[brain_path_vis] if brain_frame else None)
                preds_path_vis, reg_out_vis = model(img_path_vis)
                pred_path_idx_vis = torch.argmax(preds_path_vis, dim=1)[0].cpu()

                path_img_cpu = img_path_vis[0, 0].cpu()
                path_pred_cpu = pred_path_idx_vis
                clean_seg_cpu = target_cpu[0]
                clean_pred_cpu = pred_idx[0]

                # Raw-fetched but RESCALED regression ground truth, matching
                # EXACTLY what the training loop's own reg_target_path now
                # computes -- tanh((img_path - img_clean) / (p99-p1)), not
                # the plain difference. Fetched fresh here (clean_img_raw is
                # already the true unnormalized clean image; image_path
                # needs the same raw fetch img_path_vis already does
                # normalized, alongside it).
                p_lo_vis, p_hi_vis = frame_percentiles(img_path_raw_vis.clamp(min=0), brain_path_vis)
                scale_vis = (p_hi_vis - p_lo_vis).clamp(min=1e-6)
                reg_gt_vis = torch.tanh((img_path_raw_vis - clean_img_raw) / scale_vis)[0, 0].cpu()

                # Model's own regression estimates, both already computed:
                # reg_out[0] from THIS step's main forward pass (index 0 is
                # always the clean stream, regardless of whether pathology
                # was used this step) and reg_out_vis from the fresh
                # pathological-stream pass just above.
                reg_est_clean_cpu = reg_out[0, 0].float().cpu()
                reg_est_path_cpu = reg_out_vis[0, 0].float().cpu()
                # Isolates the model's learned sensitivity to pathology from
                # whatever baseline bias the regression head has even on
                # healthy input -- ideally close to reg_gt_vis if (2,2) is
                # accurately tracking real pathology and (1,2) is accurately
                # near zero.
                reg_est_diff_cpu = reg_est_path_cpu - reg_est_clean_cpu

                # diff_seg is deliberately NOT used for slice selection
                # anymore: the pathological stream's classification target is
                # now seg_clean_idx too (see the training loop's target
                # construction), identical to the clean stream's target by
                # construction -- a diff between them is always exactly
                # zero, and would always pick the volume's geometric center
                # rather than where anything interesting is. The regression
                # ground truth is the meaningful "where did something
                # happen" signal now.
                # diff_pred survives: predictions on clean vs pathological
                # input CAN differ even though their targets don't, and
                # where they do is worth seeing.
                diff_pred = (clean_pred_cpu != path_pred_cpu).float()

                def per_slice_score(diff_3d):
                    s = diff_3d.sum(dim=(0, 1)).float()
                    return s / s.max() if s.max() > 0 else s

                combined_score = per_slice_score(reg_gt_vis.abs()) + per_slice_score(diff_pred)
                slice_idx = combined_score.argmax().item() if combined_score.max() > 0 else clean_cpu.shape[-1] // 2

                mode_name = case['pathology_metadata'].get('mode_name', 'unknown')
                k_rose_str = f"k_rose={case.get('k_rose', 'N/A'):.2f}"
                n_changed_pred = int(diff_pred[:, :, slice_idx].sum().item())
                reg_gt_mean_abs = reg_gt_vis[:, :, slice_idx].abs().mean().item()
                log_print(f"  [viz step {i}] mode={mode_name} ({k_rose_str}) -> slice {slice_idx} "
                          f"(largest combined regression-magnitude+pred-inconsistency)  "
                          f"pred_voxels_changed={n_changed_pred}  reg_gt_mean_abs={reg_gt_mean_abs:.3f}  "
                          f"[{'CHEAP-REPEAT' if case['is_cheap_repeat'] else 'fresh draw'}]")

                # --- realized-vs-requested contrast -------------------------------
                # DISABLED, not just skipped silently: measure_realized_contrast/
                # measure_realized_texture_ratio/measure_discriminability all key
                # off LESION_COMPACT_IDX appearing in the segmentation target to
                # know WHERE to measure -- the classification target is now seg_clean_idx and
                # will never contain that class again, so all three would
                # deterministically return None forever if called. That's a
                # parallel version of the exact silent-failure the validation
                # harness had last message. Needs its own redesign working off
                # the regression map directly (e.g. measuring realized_k as
                # reg_gt_vis's own magnitude relative to k_rose, rather than
                # gating on a label that no longer exists) -- not attempted
                # here, live_diagnostics.py is untouched by this refactor so far.

                # --- whole-volume class collapse ----------------------------------
                # clean_pred_cpu/path_pred_cpu are the full 3D predictions, not the
                # single displayed slice -- this catches a collapse anywhere in the
                # volume, not just at slice_idx.
                clean_dom_idx, clean_dom_frac, _ = diag.dominant_class_fraction(clean_pred_cpu, FS_N_LABELS)
                path_dom_idx, path_dom_frac, _ = diag.dominant_class_fraction(path_pred_cpu, FS_N_LABELS)
                if path_dom_frac > clean_dom_frac + 0.15:
                    dom_name = diag.compact_idx_to_name(path_dom_idx, LABEL_LIST_SEGMENTATION, FREESURFER_LABELS)
                    log_print(f"  [viz step {i}] COLLAPSE WARNING: pathological prediction is "
                              f"{path_dom_frac:.1%} class '{dom_name}' (clean baseline {clean_dom_frac:.1%})")

                # TWO separate scales, not one shared across all four panels.
                # Ground truth can have a genuinely large peak (a strong
                # k_rose draw), and sharing that scale with the model's own
                # estimates would wash out a real-but-smaller learned signal
                # to near-white regardless of whether the estimate has any
                # structure at all -- exactly the ambiguity that makes these
                # panels hard to read. The three ESTIMATE panels still share
                # ONE scale with each other, so clean vs pathological vs
                # diff stay comparable to each other, just not to ground
                # truth's potentially much larger range.
                reg_vmax_gt = max(reg_gt_vis.abs().max().item(), 1e-6)
                reg_vmax_est = max(reg_est_clean_cpu.abs().max().item(),
                                   reg_est_path_cpu.abs().max().item(),
                                   reg_est_diff_cpu.abs().max().item(), 1e-6)

                # Custom dark-centered diverging colormap: matplotlib's
                # built-in diverging maps (seismic, RdBu, coolwarm) are all
                # white at their center, which renders a mostly-near-zero
                # regression map as a mostly-blank white panel -- jarring
                # next to the dark 'gray'/FS_CMAP panels either side of it.
                # Blue (negative) - black (zero) - red (positive) instead,
                # so zero matches the dark background everywhere else in
                # the figure and only real deviation from zero stands out.
                dark_diverging = LinearSegmentedColormap.from_list(
                    "dark_diverging", ["#3b82f6", "#000000", "#ef4444"])

                fig, ax = plt.subplots(3, 3, figsize=(15, 15))
                fig.patch.set_facecolor("black")
                grid_spec = [
                    [(clean_cpu, 'gray', "Clean Input", None, None),
                     (reg_est_clean_cpu, dark_diverging, "Regression Est. (clean, ~0 expected)", -reg_vmax_est, reg_vmax_est),
                     (clean_pred_cpu, FS_CMAP, "Segmentation Prediction", 0, FS_N_LABELS - 1)],
                    [(path_img_cpu, 'gray', "Pathological Input", None, None),
                     (reg_est_path_cpu, dark_diverging, "Regression Est. (pathological)", -reg_vmax_est, reg_vmax_est),
                     (path_pred_cpu, FS_CMAP, "Segmentation Prediction (want ~healthy)", 0, FS_N_LABELS - 1)],
                    [(reg_gt_vis, dark_diverging, "Regression Ground Truth (img diff)", -reg_vmax_gt, reg_vmax_gt),
                     (reg_est_diff_cpu, dark_diverging, "Regression Est. Diff (path - clean)", -reg_vmax_est, reg_vmax_est),
                     (clean_seg_cpu, FS_CMAP, "True Healthy Label (both streams' target)", 0, FS_N_LABELS - 1)],
                ]
                for row_idx, row in enumerate(grid_spec):
                    for col_idx, (data, cmap, title, vmin, vmax) in enumerate(row):
                        a = ax[row_idx, col_idx]
                        a.imshow(data[:, :, slice_idx], cmap=cmap, vmin=vmin, vmax=vmax)
                        a.set_title(title, fontsize=10, color="white")
                        a.axis('off')
                fig.suptitle(f"Step {i}  |  mode: {mode_name}  |  {k_rose_str}  |  slice {slice_idx}  |  {'CHEAP-REPEAT' if case['is_cheap_repeat'] else 'fresh draw'}", fontsize=12, color="white")

            plt.tight_layout()
            # facecolor passed explicitly: savefig's own default facecolor
            # does not reliably follow fig.set_facecolor() across matplotlib
            # versions, so the black background set above could otherwise
            # silently revert to white in the saved file specifically.
            plt.savefig(os.path.join(progress_dir, f"step_{i:05d}.png"),
                       facecolor=fig.get_facecolor())
            plt.close(fig)

    # Checkpointing
    if i % 5000 == 0:
        ckpt_path = os.path.join(ckpt_dir, f"synthseg_{i:05d}.pth")
        _ck = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            # log_sigma_seg/log_sigma_reg are standalone tensors, not
            # attributes of model -- NOT covered by model.state_dict(),
            # and optimizer.state_dict() only saves Adam's internal moment
            # estimates for them, not their own learned values. Without
            # this, resuming from a checkpoint would silently reset the
            # learned task balance back to equal weighting every time.
            "task": args.task,
            "log_sigma_seg": log_sigma_seg.detach().cpu(),
            "log_sigma_reg": log_sigma_reg.detach().cpu(),
            "step": i,
            "rng_state": _rng_state_dict(),
        }
        if _ema is not None:
            _ck["ema_state_dict"] = {k: v.detach().cpu() for k, v in _ema.items()}
        torch.save(_ck, ckpt_path)
        log_print(f"Saved Checkpoint: {ckpt_path}")
        if _ema is not None:
            _ema_path = ckpt_path.replace(".pth", "_ema.pth")
            torch.save({"model_state_dict": {**model.state_dict(), **_ema}, "step": i, "ema_decay": args.ema}, _ema_path)
            log_print(f"Saved EMA weights: {_ema_path}")

RUNNING_DIAG.final_report(log_print)
log_print(f"TRAINING COMPLETE FOR {RUN_NAME}")
log_file.close()
