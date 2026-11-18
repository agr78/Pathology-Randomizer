"""Downstream tasks on the pathology generator: registration and super-resolution.

Same generator, same UNet, no real pathology anywhere. One task per run:

  --register       SynthMorph-style. Two independent generator draws; moving =
                   draw A (pathological with --pathology_prob), fixed = draw B
                   (clean). SynthSegUNet(in=2, out=3) predicts a stationary
                   velocity field, integrated by scaling-and-squaring into a
                   displacement. Loss = 1 - soft Dice of the warped healthy
                   label map against the fixed one + lambda * |grad v|^2.
  --super_resolve  SynthSR-style. Input = the generator's degraded image
                   (its own resolution sampler: blur, slice thickness,
                   resampling, noise); target = the pre-degradation image the
                   generator now exposes (image_*_hr). SynthSegUNet(in=1,
                   out=1) predicts the residual; loss = L1 on the brain.

Checkpoints, logs and naming follow train.py:
  <out_dir>/<task>[_healthy][_<tag>]_<hash>_out/log_<ts>/checkpoints/<task>_NNNNN.pth

2026-10-14 (whole-network fine-tune protocol from the 300k trunks; every default keeps the earlier behaviour):
  --attention / --attention_dense / --attention_identity_init  gates on a gate-less --init_from trunk, identity-open as train.py
  --dark_csf P            generator.DARK_CSF_PROB as train.py (module global; the prefetcher is a thread, so it reaches the producer)
  --aug_rotation/_scaling/_shear  residual affine augmentation of registration draws (generator units; 0/0/0 = frame-locked pairs)
  --resume CKPT           checkpoints now carry optimizer, scaler, step, args and RNG (~153 MB each: weights + two Adam moments, 3x the
                          51 MB weight-only files; 20 per 100k run); resume continues in the same run directory
  --strip_prob P          super-resolution only: that fraction of draws is skull-stripped (input and target, before normalization,
                          one voxel of rim). The generator always renders a whole head while BraTS ships skull-stripped, and on that
                          unseen input the net over-scales its residual about 2x; the cause is the
                          missing tissue class, not the intensity frame (tested by forcing the whole-head frame)
  --identity_prob P / --identity_weight W  registration only: on a fraction P of steps the moving draw's clean rendering is also
                          registered to ITSELF and W x mean |u|^2 (vox^2) is added, the one pair whose true field is known to be zero
                          (2026-10-14: the label-Dice loss alone admits an input-independent field that contracts each fixed structure
                          to a point; a registration self-test measures it, validation now reports "identity |u|")
  --bn_mode batch|running validation BN convention (batch = compare_datasets.bn_batch_mode, the detection model's readout; FeatureHeadNet unaffected)
  validation diagnostics: registration |u|, 12-dof affine share, net translation, lesion sensitivity; SR lesion PSNR, target clip fraction
  load_task_model(..., bn_mode='batch') returns whole-network evaluation nets under batch statistics (the same convention as validation; a FeatureHeadNet is unaffected)
"""
import argparse, math
import os
import random
import subprocess
import time
from datetime import datetime

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from generator_downstream import paired_pathology_generator   # the generator build the downstream fine-tunes ran with (B-spline zoom-back, intra-subject pairs); generator.py is the detection build
from networks import SynthSegUNet, detect_attention, detect_attention_dense
from prefetch import Prefetcher
from util import make_mode_sampler
import pathology_primitives

LABEL_LIST_SEGMENTATION = [0, 11, 12, 13, 16, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                           43, 44, 46, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 17, 47, 49, 51,
                           53, 55, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 48, 50,
                           52, 54, 56]
N_LABELS = len(LABEL_LIST_SEGMENTATION)
EXTRACEREBRAL = list(range(8, 20))          # compact idx of ids 34..46 (extracerebral block)
LESION_IDX = 7                              # compact idx of the synthetic lesion (id 33) in seg_path_idx (2026-10-14)
RESUME_ARCH = ("register", "super_resolve", "feature_input", "attention", "attention_dense", "sr_calibrate", "sr_uncertainty")   # must match the checkpoint's args on --resume


def git_id():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass
    try:
        return open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "BASE_COMMIT")).read().split()[0][:7]
    except Exception:
        return "nogit"


# ---------------------------------------------------------------- geometry
def base_grid(shape, device, dtype):
    """Identity sampling grid for grid_sample(align_corners=False): voxel i
    of s sits at (2i + 1) / s - 1, not linspace(-1, 1, s) (which shrinks by
    one voxel and is not the identity)."""
    ax = [(2 * torch.arange(s, device=device, dtype=dtype) + 1) / s - 1 for s in shape]
    gz, gy, gx = torch.meshgrid(*ax, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1)[None]


def warp(vol, flow, mode="bilinear"):
    """flow: (B,3,D,H,W) displacement in normalized grid units (x,y,z order)."""
    grid = base_grid(vol.shape[2:], vol.device, flow.dtype) + flow.permute(0, 2, 3, 4, 1)
    return F.grid_sample(vol, grid, mode=mode, padding_mode="border", align_corners=False)


def integrate_svf(v, steps=7):
    """Scaling and squaring: displacement = exp(v)."""
    phi = v / (2 ** steps)
    for _ in range(steps):
        phi = phi + warp(phi, phi)
    return phi


def vox_scale(flow):
    """Per-channel factor taking a normalized (x,y,z) flow to voxel units, shaped (1,3,1,1,1)."""
    D, H, W = flow.shape[2:]
    return torch.tensor([W / 2, H / 2, D / 2], device=flow.device, dtype=flow.dtype).view(1, 3, 1, 1, 1)


def grad_penalty(v):
    """Mean squared finite-difference gradient of the field, in voxel units."""
    u = v * vox_scale(v)
    return sum(((u[:, :, 1:] - u[:, :, :-1]) ** 2).mean() +
               ((u[:, :, :, 1:] - u[:, :, :, :-1]) ** 2).mean() +
               ((u[:, :, :, :, 1:] - u[:, :, :, :, :-1]) ** 2).mean() for _ in [0])


def jacobian_fold_fraction(flow):
    """Fraction of voxels where the warp folds (det J <= 0)."""
    u = flow * vox_scale(flow)                              # voxel units, channels (x,y,z)
    # gradients along z (dim2), y (dim3), x (dim4)
    dz = torch.gradient(u, dim=2)[0]; dy = torch.gradient(u, dim=3)[0]; dx = torch.gradient(u, dim=4)[0]
    # J[i][j] = d(phi_i)/d(x_j) with phi = x + u
    J = torch.stack([torch.stack([dx[:, 0] + 1, dy[:, 0], dz[:, 0]], -1),
                     torch.stack([dx[:, 1], dy[:, 1] + 1, dz[:, 1]], -1),
                     torch.stack([dx[:, 2], dy[:, 2], dz[:, 2] + 1], -1)], -2)
    return float((torch.det(J) <= 0).float().mean())


def global_field(flow, region):
    """Copy of eval/register_eval.global_field (2026-10-14; that module imports this one): the resting global part of the field over
    `region` (fixed brain). Returns (u_affine_share, u_translation): the fraction of sum|u|^2 explained by the least-squares 12-dof affine
    u ~ A p + b of the voxel-unit displacement on voxel coordinates p (NaN for a zero field), and the norm of the mean displacement (vox)."""
    u = (flow * vox_scale(flow))[0][:, region].T.double()   # (N,3) vox
    p = torch.nonzero(region).double(); X = torch.cat([p - p.mean(0), torch.ones(len(p), 1, device=p.device, dtype=p.dtype)], 1)
    beta = torch.linalg.solve(X.T @ X, X.T @ u); tot = float((u ** 2).sum())
    share = float(1 - ((u - X @ beta) ** 2).sum() / tot) if tot > 0 else float("nan")
    return share, float(u.mean(0).norm())


def soft_dice_labels(warped_onehot, fixed_onehot, smooth=1.0):
    inter = (warped_onehot * fixed_onehot).sum(dim=(2, 3, 4))
    den = warped_onehot.sum(dim=(2, 3, 4)) + fixed_onehot.sum(dim=(2, 3, 4))
    return (2 * inter + smooth) / (den + smooth)            # (B, L)


def label_dice_chunked(moving_idx, fixed_idx, flow, labels, chunk=8):
    """Soft Dice per label of the warped moving one-hot vs the fixed one-hot;
    idx maps are (D,H,W); a few labels are warped at a time to bound memory.
    Returns the mean over labels."""
    dices = []
    for i in range(0, len(labels), chunk):
        lab = labels[i:i + chunk]
        m = torch.stack([(moving_idx == l).float() for l in lab], 0)[None]   # (1,c,D,H,W)
        f = torch.stack([(fixed_idx == l).float() for l in lab], 0)[None]
        dices.append(soft_dice_labels(warp(m, flow), f))
    return torch.cat(dices, 1).mean()


def normalize(x):
    """train.py's normalize_per_sample for one volume: clip to [p1, p99] of
    the foreground, rescale to [0, 1]. Returns the scale for reuse."""
    fg = x[x > 0]
    if fg.numel() == 0:
        return x, (0.0, 1.0)
    lo, hi = torch.quantile(fg, torch.tensor([0.01, 0.99], device=x.device, dtype=x.dtype))
    return ((x.clamp(min=lo, max=hi) - lo) / (hi - lo).clamp(min=1e-6)), (lo, hi)


def apply_scale(x, lo, hi):
    return ((x.clamp(min=lo, max=hi) - lo) / (hi - lo).clamp(min=1e-6))


def psnr(a, b, mask):
    mse = ((a - b) ** 2)[mask].mean().clamp(min=1e-12)
    return float(10 * torch.log10(1.0 / mse))


def brain_mask(lab):
    """Brain voxels of a compact-index label map: labelled and not extracerebral."""
    return (lab > 0) & ~torch.isin(lab, torch.tensor(EXTRACEREBRAL, device=lab.device))


# ---------------------------------------------------------------- BatchNorm convention (2026-10-14)
class bn_eval:
    """Context manager for validation / evaluation: model.eval(), and with bn_mode == 'batch' compare_datasets.bn_batch_mode (every BN layer
    back in train mode with momentum 0, so each volume is normalised by its own statistics and the running buffers stay untouched) -- the
    detection model's readout. bn_batch_mode leaves momentum at 0 permanently, which would freeze the running statistics for the rest of
    training, so every BN's momentum is restored on exit, then model.train(). FeatureHeadNet: unaffected by bn_mode, as in load_task_model
    (2026-10-14 review): its train() override keeps the trunk in batch mode and the head stays on running statistics, as before."""

    def __init__(self, model, bn_mode):
        if bn_mode not in ("batch", "running"):
            raise ValueError(f"bn_mode must be 'batch' or 'running', got {bn_mode!r}")
        self.model, self.bn_mode = model, bn_mode

    def __enter__(self):
        self.momenta = [(m, m.momentum) for m in self.model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        self.model.eval()
        if self.bn_mode == "batch" and not isinstance(self.model, FeatureHeadNet):   # whole-network nets only, as load_task_model
            from compare_datasets import bn_batch_mode
            bn_batch_mode(self.model)
        return self.model

    def __exit__(self, *exc):
        for m, mom in self.momenta:
            m.momentum = mom
        self.model.train()
        return False


# ---------------------------------------------------------------- feature-input nets (2026-10-09)
class FeatureHeadNet(nn.Module):
    """The frozen detection trunk as the feature extractor, run per image exactly as at inference (its own frame, batch-statistics
    BN), and a trainable SynthSegUNet head on the concatenated 24-channel d1 features of the n input images (the tensor both
    detection heads read). The task network consumes what the detection model learned -- the anatomy posterior under a lesion and
    the deviation -- instead of raw intensities, so pathology robustness is inherited from the trunk rather than re-learned through
    a duplicated first convolution. Checkpoints carry the trunk (self-contained) and a marker buffer; see load_task_model."""

    def __init__(self, trunk, n_images, out_channels):
        super().__init__()
        self.trunk = trunk
        for q in self.trunk.parameters():
            q.requires_grad_(False)
        self.c = self.trunk.final_conv.in_channels
        self.head = SynthSegUNet(in_channels=n_images * self.c, out_channels=out_channels)
        self.register_buffer("feature_input_flag", torch.tensor([float(n_images)]))
        self._feat = {}
        self.trunk.final_conv.register_forward_hook(lambda m, i, o: self._feat.__setitem__("d1", i[0]))
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.trunk.eval()
        from compare_datasets import bn_batch_mode
        bn_batch_mode(self.trunk)          # the deployed readout: BN on the statistics of the image being inferred
        return self

    @property
    def final_conv(self):
        return self.head.final_conv

    def forward(self, x):
        sh = x.shape[2:]; pads = [(32 - s % 32) % 32 for s in sh]
        xp = F.pad(x, (0, pads[2], 0, pads[1], 0, pads[0]))
        feats = []
        for i in range(xp.shape[1]):
            with torch.no_grad():
                self.trunk(xp[:, i:i + 1])
            feats.append(self._feat.pop("d1").float())
        out = self.head(torch.cat(feats, 1))
        return out[:, :, :sh[0], :sh[1], :sh[2]]


def feature_head_from_state(sd, out_channels, device):
    """Rebuild a FeatureHeadNet from its own checkpoint state (trunk keys + marker buffer); weights are loaded by the caller."""
    from networks import model_from_state
    trunk = model_from_state({k[len("trunk."):]: v for k, v in sd.items() if k.startswith("trunk.")}, device)
    return FeatureHeadNet(trunk, int(sd["feature_input_flag"].item()), out_channels).to(device)


def load_task_model(path, device, out_channels=3, bn_mode="batch"):
    """Load a downstream checkpoint of either kind: the two-channel SynthSegUNet, or a FeatureHeadNet (marker buffer present).
    bn_mode (2026-10-14): 'batch' = compare_datasets.bn_batch_mode on the loaded net, every BatchNorm normalises the volume being
    inferred by its own statistics (momentum 0; the convention of the detection readout and of the fine-tune validation, and what
    the single-volume training batches saw); 'running' = model.eval() only, the running averages (the behaviour before this flag:
    every eval/register_eval*.py and eval/sr_eval.py row scored before 2026-10-14 used running statistics, so the default changes
    the evaluation of whole-network nets). A FeatureHeadNet is unaffected by bn_mode (2026-10-14 review): trunk in batch mode (its
    train() override), head on running statistics, as every feature-head row before this flag."""
    ck = torch.load(path, map_location="cpu", weights_only=False); sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    if "feature_input_flag" in sd:
        m = feature_head_from_state(sd, out_channels, device)
    else:
        m = SynthSegUNet(in_channels=sd["enc1.conv.0.weight"].shape[1], out_channels=out_channels, attention=detect_attention(sd),
                         attention_dense=detect_attention_dense(sd)).to(device)
        if "sr_uncertainty_marker" in sd: m.register_buffer("sr_uncertainty_marker", torch.ones(1, device=device))   # 2026-11-09: --sr_uncertainty head
    m.load_state_dict(sd); m.eval()
    if bn_mode == "batch":
        if not isinstance(m, FeatureHeadNet):   # whole-network nets only (2026-10-14 review)
            from compare_datasets import bn_batch_mode
            bn_batch_mode(m)
    elif bn_mode != "running":
        raise ValueError(f"bn_mode {bn_mode!r}: expected 'batch' or 'running'")
    return m


# ---------------------------------------------------------------- model (2026-10-14: factored out of main for the CPU tests)
def task_channels(args):
    # 2026-11-07 --sr_calibrate: a SECOND super-resolution output channel, the per-voxel AMPLITUDE of the
    # residual. Measured motivation: inside real lesions the residual correlates with the needed correction
    # at rho 0.32-0.37 against 0.51 outside, yet its MAGNITUDE is identical in both regions, so the net is
    # overconfident rather than blind. The optimal single scalar there is 0.43 and would turn -0.35 dB
    # against tricubic into +0.7 dB, about a decibel of headroom. Learning the amplitude against the SAME
    # loss puts that scaling inside the network, reading only the input, instead of an external gate that
    # needed the target. Behind a flag: out_channels stays 1 without it, so every existing checkpoint
    # still loads and every existing arm is bit-unchanged.
    if args.register: return (2, 3)
    # --sr_uncertainty: a second channel again, but the LOG-SCALE b of a per-voxel Laplace likelihood on the
    # residual's error; trained by the negative log-likelihood |e|/b + log b, so the net states where its correction is unreliable.
    # The prediction itself is the plain residual (no scaling at inference); the scale map is written out by the scorer.
    return (1, 2) if (getattr(args, 'sr_calibrate', False) or getattr(args, 'sr_uncertainty', False)) else (1, 1)


def build_model(args, init_state=None, device="cpu", log=print, arch_state=None):
    """Construct the task net and warm-start it. Gates: --attention / --attention_dense OR whatever the architecture-defining state
    carries (arch_state, default init_state: a gated checkpoint keeps its gates, as before); --attention_identity_init opens FRESH gates
    (psi bias 8, weight 0) so a gate-less trunk forked with --attention starts as the network it was (mirrors train.py). With init_state
    every matching tensor is copied (the one-channel first conv duplicated and halved for the two-image input), final_conv is zeroed
    (identity start), head / regression / marker / unmatched keys are skipped and only gate tensors absent from the checkpoint stay fresh.
    Returns (model, info) with info = dict(loaded, skipped, fresh_gates, attention, attention_dense)."""
    in_ch, out_ch = task_channels(args)
    ref = arch_state if arch_state is not None else init_state
    attention = bool(args.attention or (ref is not None and detect_attention(ref)))
    dense = bool(args.attention_dense or (ref is not None and detect_attention_dense(ref)))
    if dense and not attention:
        raise SystemExit("--attention_dense needs --attention")
    if args.attention_identity_init and not attention:
        log("WARNING: --attention_identity_init without gates (no --attention, gate-less checkpoint): ignored")
    model = SynthSegUNet(in_channels=in_ch, out_channels=out_ch, attention=attention, attention_dense=dense,
                         attention_identity_init=args.attention_identity_init).to(device)
    # start from the identity: zero the last conv so the first flow / residual is zero
    torch.nn.init.zeros_(model.final_conv.weight); torch.nn.init.zeros_(model.final_conv.bias)
    # with --sr_calibrate, bias the amplitude channel so sigmoid(b) ~ 0.98 at step 0: training then STARTS
    # from the uncalibrated behaviour and only learns where to pull the amplitude down. Zeroing it instead
    # would start every residual at half strength, a different arm from the one we are improving.
    if getattr(args, 'sr_calibrate', False) and model.final_conv.bias.shape[0] >= 2:
        with torch.no_grad(): model.final_conv.bias[1] = 4.0
    if getattr(args, 'sr_uncertainty', False) and model.final_conv.bias.shape[0] >= 2:
        # log b starts at -3 (b ~ 0.05, the order of the residual's error in the normalised frame), so the NLL is a scaled L1 from
        # step 0 rather than a constant-plus-log; a marker buffer rides in the state dict so a loader can tell this head from --sr_calibrate.
        with torch.no_grad(): model.final_conv.bias[1] = -3.0
        model.register_buffer('sr_uncertainty_marker', torch.ones(1, device=device))
    info = dict(loaded=0, skipped=[], fresh_gates=[], attention=attention, attention_dense=dense)
    if init_state is not None:
        own = model.state_dict()
        for k, v in init_state.items():
            if k not in own or k.startswith("final_conv"):
                info["skipped"].append(k); continue
            if k == "enc1.conv.0.weight" and v.shape[1] == 1 and own[k].shape[1] == 2:
                v = v.repeat(1, 2, 1, 1, 1) / 2.0      # one image channel -> the (moving, fixed) pair: each image gets half the weight
            if v.shape != own[k].shape:
                info["skipped"].append(k); continue
            own[k] = v.to(own[k].dtype); info["loaded"] += 1
        model.load_state_dict(own)
        info["fresh_gates"] = [k for k in own if k.startswith("att") and k not in init_state]
        sk = info["skipped"]
        log(f"init_from: {info['loaded']} tensors loaded, {len(sk)} skipped (head/regression/unmatched: {sk[:4]}{'...' if len(sk) > 4 else ''}), "
            f"{len(info['fresh_gates'])} fresh gate tensors{' (identity-open)' if info['fresh_gates'] and args.attention_identity_init else ''}; "
            f"attention {attention}{' dense' if dense else ''}; output head zero")
    return model, info


# ---------------------------------------------------------------- checkpoints and resume (2026-10-14)
def rng_state():
    """As train.py._rng_state_dict (numpy's MT key as a list: weights_only-safe)."""
    st = np.random.get_state()
    return {"python": random.getstate(), "numpy": (st[0], st[1].tolist(), int(st[2]), int(st[3]), float(st[4])),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def rng_restore(st, log=print):
    py = st["python"]; random.setstate((py[0], tuple(py[1]), py[2]))
    n = st["numpy"]; np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(st["torch"].cpu())
    n_saved, n_here = len(st["cuda"]), (torch.cuda.device_count() if torch.cuda.is_available() else 0)
    for d in range(min(n_saved, n_here)):
        torch.cuda.set_rng_state(st["cuda"][d].cpu(), device=d)
    if n_saved != n_here:
        log(f"WARNING: CUDA RNG state saved for {n_saved} device(s), {n_here} visible; restored {min(n_saved, n_here)}")


def save_checkpoint(path, model, opt, scaler, step, args, ema=None):
    """Full training state: model, optimizer, scaler (fp16 only), step, args and the RNG streams (torch, numpy, python, cuda).
    ~153 MB per file for the 12.7M-parameter nets (weights + two Adam moments) against 51 MB weight-only before 2026-10-14."""
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                "step": int(step), "args": vars(args), "rng": rng_state(), **({"ema_state_dict": {k: v.detach().cpu() for k, v in ema.items()}} if ema else {})}, path)


def check_resume(ck, args, path="<ckpt>"):
    """Refuse to resume a checkpoint without optimizer state (pre-2026-10-14: use --init_from) or whose task / architecture flags differ."""
    if not (isinstance(ck, dict) and "optimizer_state_dict" in ck and "step" in ck):
        raise SystemExit(f"--resume {path}: no optimizer state / step (pre-2026-10-14 checkpoint); warm-start it with --init_from instead")
    old = ck.get("args", {})
    diff = {k: (old.get(k, False), getattr(args, k)) for k in RESUME_ARCH if bool(old.get(k, False)) != bool(getattr(args, k))}
    if diff:
        raise SystemExit(f"--resume {path}: task / architecture flags differ from the checkpoint's (checkpoint, now): {diff}")


def load_resume(ck, model, opt, scaler, log=print, restore_rng=True):
    """Load model + optimizer (+ scaler, + RNG when present) from a full-state checkpoint; returns its step."""
    model.load_state_dict(ck["model_state_dict"]); opt.load_state_dict(ck["optimizer_state_dict"])
    if scaler.is_enabled() and ck.get("scaler_state_dict"):
        scaler.load_state_dict(ck["scaler_state_dict"])
    if restore_rng and "rng" in ck:
        rng_restore(ck["rng"], log)
    return int(ck["step"])


# ---------------------------------------------------------------- data
def affine_kwargs(args):
    """Registration draws (2026-10-14): residual affine augmentation in the generator's own units -- max_rotation in DEGREES (each axis
    U[-r, r]), max_scaling and max_shear as FRACTIONS (scale 1 +- s, shear +- h per axis; generator.py 328-330), drawn independently for
    the moving and the fixed subject (the pair's residual is their composition; centre_on_brain removes the translation). Defaults 0/0/0 = the
    frame-locked pairs trained until now: with the generator's per-draw default (+-15 deg, 0.2, 0.2) two draws overlap at Dice ~0.02
    (measured on validation_set_v2 pairs) and a deformable net cannot start from that. SR passes nothing and keeps the defaults."""
    if args.super_resolve:
        return {}
    return dict(max_rotation=args.aug_rotation, max_shear=args.aug_shear, max_scaling=args.aug_scaling)


def generator_kwargs(args, device):
    """Every keyword of paired_pathology_generator for this task (inspectable without running the generator)."""
    return dict(
        datadir=args.data_dir, mode_sampler=make_mode_sampler(restricted=False), **affine_kwargs(args),
        label_list_segmentation=LABEL_LIST_SEGMENTATION, n_neutral_labels=20,
        label_list_normalization=[1, 18], sides='both', labels_to_kill=[45],
        list_to_kill_photo=[], list_to_kill_photo_inc_ventricles=[], flipping=True,
        mix_real_images=False, susceptibility=False, couple_appearance=False,
        complex_resample=False, gibbs_ringing_filter=False, generate_pathology=True,
        minimal_io=True, compact_labels=True, random_kill_labels=list(range(34, 47)),
        random_kill_prob=0.5, random_kill_frac=(1.0, 1.0), device=device,
        healthy_only=args.healthy, k_rose_sampler=lambda: float(np.random.uniform(1.0, 2.5)),
        mode_weights=(0.23, 0.0, 0.0, 0.385, 0.385),
        # The generator yields each subject cheap_repeat_k times in a row
        # (re-perturbed). Registration needs two DIFFERENT subjects per pair,
        # so repeats are disabled there; consecutive draws would otherwise be
        # the same anatomy and the task collapses to Dice ~1 at the identity.
        cheap_repeat_k=1 if args.register else 3,
        # 2026-10-27 --intra_prob: a fraction of registration pairs is one subject (same draw, same lesion) in two poses; the
        # generator queues the twin only after an even-indexed draw so it always arrives as `b` (register_step). No loss change.
        intra_pair_prob=(args.intra_prob if args.register else 0.0),
        # 2026-10-16: the generator's own nonlinear warp, exposed so the registration pair's shape difference can be made
        # larger. Defaults (scale 0.03-0.06, std_max 4) are the generator's. The ventricle test: a real ventricle mismatch is a
        # STRUCTURED local volume change, while this warp is smooth and random -- raising its amplitude says whether more of the
        # same kind of deformation closes the CSF gap to SynthMorph (0.29 vs 0.44) or whether the KIND is what is missing.
        # 2026-10-31: the zoom-back interpolation of the simulated low-resolution acquisition. False (the generator's default,
        # never once enabled before today) uses myzoom_torch = LINEAR; True uses interpol.resize(..., interpolation=3,
        # bound='dct2', prefilter=True) at generator.py:2725 = cubic B-spline, i.e. tricubic. Exposed as a FLAG rather than a
        # changed default on purpose: registration shares this generator entry point, so a flipped default would silently alter
        # the running registration arms on their next auto-resume.
        bspline_zooming=args.bspline_zoom,
        **({} if args.nonlin_std is None else dict(nonlin_std_max=args.nonlin_std)),
        **({} if args.nonlin_scale_max is None else dict(nonlin_scale_max=args.nonlin_scale_max)))


def build_generator(args, device):
    # Registration pairs must share an affine frame (see affine_kwargs); brains are also centred on their centroid (centre_on_brain).
    return paired_pathology_generator(**generator_kwargs(args, device))


def set_dark_csf(p, log=print):
    """--dark_csf as train.py 681-684 (2026-10-14): generator.DARK_CSF_PROB is a module global read per rendered image (generator.py 593,
    2354), so it must be set BEFORE the generator / prefetcher is built. prefetch.Prefetcher is a THREAD in this interpreter (same module
    object), so the global set here reaches the producer; there is no process-based prefetch path to handle."""
    if p > 0:
        import generator as _gen
        _gen.DARK_CSF_PROB = float(p)
        log(f"dark_csf on (p={p})")


def centre_on_brain(case):
    """Integer-shift every volume of a case so the brain centroid sits at the
    volume centre (torch.roll; the border is background)."""
    lab = case["seg_clean_idx"]
    brain = brain_mask(lab)
    if brain.sum() == 0:
        return case
    idx = torch.nonzero(brain).float().mean(0)
    c = (torch.tensor(lab.shape, device=lab.device, dtype=torch.float32) - 1) / 2
    shift = [int(round(float(c[i] - idx[i]))) for i in range(3)]
    return {k: (torch.roll(v, shifts=shift, dims=(0, 1, 2)) if torch.is_tensor(v) else v) for k, v in case.items()}   # flags pass through


def case_tensors(case, device):
    out = {}
    for k in ("image_clean", "image_path", "image_clean_hr", "image_path_hr"):
        if k in case:
            out[k] = case[k].detach().to(device, non_blocking=True).float()
    for k in ("seg_clean_idx", "seg_path_idx"):
        out[k] = case[k].detach().to(device, non_blocking=True).long()
    out["intra_pair"] = bool(case.get("intra_pair", False))   # 2026-10-27: this draw is the previous draw's subject in a second pose
    return out


# ---------------------------------------------------------------- tasks
def _autocast(x, args):
    return torch.amp.autocast('cuda', dtype=(torch.bfloat16 if args.amp == "bf16" else torch.float16), enabled=(x.is_cuda and args.amp != "off"))


def flow_of(model, mov, fix, args):
    """Velocity -> displacement for one normalised (moving, fixed) pair; returns (v, flow)."""
    x = torch.stack([mov, fix])[None]                     # (1,2,D,H,W)
    with _autocast(x, args):
        v = model(x)
    v = v.float() * args.vel_scale
    return v, integrate_svf(v, args.int_steps)


def affine_prealign(a, b, args):
    """SynthMorph's JOINT paradigm: compose an affine with the deformation at TRAINING time, so the deformable
    stage learns on pairs that are already affinely aligned -- the condition every deployment imposes and that this task never did.

    Why: the fine-tune's pairs start at tissue Dice 0.17-0.25 and need ~4 voxel, 40 %-global corrections, while a real pair reaches
    the network at 0.53-0.57 after a 12-dof fit. Measured consequence (40 ABIDE pairs x 2 seeds): the
    field's amplitude is near-CONSTANT across pairs (CV 0.11 at 55k) and uncorrelated with how misaligned the pair is (|r| <= 0.07),
    i.e. the net applies its training-typical correction regardless; on an aligned pair that correction is pure cost (the alpha
    sweep: damage rises monotonically with amplitude, best at alpha -> 0), and it grows with training as the learned amplitude grows.
    Measured here on validation draws: centring alone starts a pair at 0.250, the moments solution at 0.321, the full 12-dof label
    fit at 0.437 -- so `moments` closes a third of the gap and `fit` most of it.

    Fits the moving draw onto the fixed one with the SAME routine the evaluation uses (eval/register_eval_pool.affine_to: soft Dice
    of the large labels, Adam, 4x-coarse grid), then resamples the moving image, its pathological rendering and its label map.
    Imported lazily: register_eval_pool imports this module."""
    if args.affine_prealign == "none": return a
    import importlib, os as _os, sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "eval"))
    P = importlib.import_module("register_eval_pool")
    A = dict(lab=a["seg_clean_idx"], img=a["image_clean"], img_path=a["image_path"])
    with torch.enable_grad():   # affine_to optimises its 12 parameters with Adam, so it needs grad even when the CALLER does not:
        #                         validate_register runs under torch.no_grad(), where loss.backward() aborted the process at the
        #                         first validation of run 10235382 (2026-10-15). The fit is independent of the model; the outputs
        #                         are detached below so nothing leaks into the caller's graph.
        out = P.affine_to(A, dict(lab=b["seg_clean_idx"]), ("img", "img_path", "lab"),
                          iters=(args.prealign_iters if args.affine_prealign == "fit" else 1))
    return dict(a, image_clean=out["img"].detach(), image_path=out["img_path"].detach(), seg_clean_idx=out["lab"].detach().long())


def register_step(model, a, b, args):
    """a = moving draw, b = fixed draw (dicts of tensors). Runs under the caller's grad mode (2026-10-14: main() re-enables grad after
    every next(gen); validation keeps no_grad)."""
    a, b = centre_on_brain(a), centre_on_brain(b)
    a = affine_prealign(a, b, args)          # the affine stage, when enabled: the net then only sees the residual
    use_path = np.random.rand() < args.pathology_prob
    # --swap_prob (2026-10-16): the pathological rendering belongs to draw `a`; swapping the ROLES of the two draws puts it on
    # the FIXED side instead. Without this the net only ever sees pathological->healthy, so registering two patients (longitudinal
    # follow-up, the obvious clinical ask) is out of distribution. At swap_prob 0.5 with pathology_prob 0.5 the four quadrants are
    # 25 % lesion on moving, 25 % lesion on fixed, 50 % neither.
    swap = args.swap_prob > 0 and np.random.rand() < args.swap_prob
    intra = bool(b.get("intra_pair", False))   # --intra_prob (2026-10-27): b is a's subject in a second pose, same lesion
    if intra: swap = False                        # the lesion then sits on BOTH sides; labels and loss are unchanged
    mov_d, fix_d = (b, a) if swap else (a, b)
    mov_img = normalize(mov_d["image_path"] if (use_path and (intra or not swap)) else mov_d["image_clean"])[0]
    fix_img = normalize(fix_d["image_path"] if (use_path and (intra or swap)) else fix_d["image_clean"])[0]
    mov_lab = mov_d["seg_clean_idx"]        # healthy anatomy of the moving subject
    fix_lab = fix_d["seg_clean_idx"]
    v, flow = flow_of(model, mov_img, fix_img, args)
    labels = [l for l in torch.unique(torch.cat([mov_lab.flatten(), fix_lab.flatten()])).tolist()
              if l != 0 and l not in EXTRACEREBRAL]
    dice = label_dice_chunked(mov_lab, fix_lab, flow, labels)
    smooth = grad_penalty(flow if args.smooth_on == "displacement" else v)   # SynthMorph: ||grad u||^2 on the integrated displacement; velocity = the VoxelMorph-diff convention this file started with
    loss = (1 - dice) + args.lambda_smooth * smooth
    ident = torch.zeros((), device=flow.device)
    if args.identity_prob > 0 and np.random.rand() < args.identity_prob:
        # Degenerate-field guard (2026-10-14). The Dice term cannot see a field that contracts every FIXED structure to a point:
        # sampling the moving image near a structure's centre returns that structure's label, so the warped label map scores well
        # without the field reading the moving image at all. Measured on the ft300k arm: registering MNI152 to ITSELF gave mean
        # |u| 3.2 vox at 5k and 6.3 vox at 10k, the same field as on every real pair, with the deep structures painted as flat
        # ellipsoids. The field is input-independent, so penalising it on ONE identical pair penalises it everywhere. Truth on an
        # identical pair is u = 0 exactly, hence a plain quadratic in voxels: mean |u|^2 ~ 50 vox^2 in the degenerate state
        # (x 0.01 = 0.5, well above the ~0.15 of Dice the strategy buys) and < 0.02 once |u| is around a voxel, where its gradient
        # also vanishes -- unlike an L1 form, which keeps pushing toward the zero field (the 2026-10-07 zero-flow failure).
        mov_c = normalize(mov_d["image_clean"])[0]
        _, flow_i = flow_of(model, mov_c, mov_c, args)
        ident = ((flow_i * vox_scale(flow_i)).norm(dim=1)[0] ** 2)[brain_mask(mov_lab)].mean()
        loss = loss + args.identity_weight * ident
    cons = torch.zeros((), device=flow.device)
    if args.consistency > 0 and use_path:
        # Pathology-invariance term (2026-10-08): the SAME moving draw without its lesion must give the same flow. Both renderings
        # come from the generator for free; the Dice term above stays on the pathological flow, this only ties the two flows
        # together (gradients through both). Measured by eval/register_shell_val.py as "lesion sensitivity" (vox).
        if swap: _, flow_c = flow_of(model, mov_img, normalize(fix_d["image_clean"])[0], args)
        else:    _, flow_c = flow_of(model, normalize(mov_d["image_clean"])[0], fix_img, args)
        cons = ((flow - flow_c) * vox_scale(flow)).norm(dim=1).mean()          # mean |u_path - u_clean| in voxels over the volume
        loss = loss + args.consistency * cons
    return loss, {"dice": float(dice), "smooth": float(smooth), "fold": jacobian_fold_fraction(flow.detach()), "cons": float(cons), "ident": float(ident), "intra": float(intra)}


def sr_step(model, a, args, use_path=None, diagnostics=False):
    """use_path=None draws the pathology rendering with prob pathology_prob (training); validation passes True / False explicitly.
    diagnostics (validation, 2026-10-14): psnr_lesion = output PSNR inside the synthetic lesion (seg_path_idx == LESION_IDX, NaN when
    absent) and clip_frac = fraction of brain target voxels outside the input's [p1, p99] window, i.e. clamped by apply_scale (the target
    clamp convention is kept as is)."""
    # The pathological draw is
    # APPENDED to the batch, as train.py:1446 does (`img = torch.cat([img_clean_norm, img_path_norm])`), instead of REPLACING the
    # clean one. Substituting made the clean-data budget depend on pathology_prob: at 0.5 over 200k steps an arm saw ~100k clean
    # steps while its pathology_prob 0 control saw 200k, so every matched pair was confounded twofold on the budget that sets
    # ~97 % of the SR metric (measured: -0.25 dB in-domain at matched TOTAL steps, +0.09 dB at matched CLEAN steps). Appending
    # keeps clean exposure at exactly one sample per step for BOTH arms, so the pair is matched by construction.
    # The strip draw is made ONCE for the step and applied to every sample, so the batch is consistently stripped or not.
    # Validation passes use_path explicitly and therefore still runs a single sample: the logged curves stay comparable.
    draws = [use_path] if use_path is not None else ([False] + ([True] if np.random.rand() < args.pathology_prob else []))
    strip_fired = args.strip_prob > 0 and np.random.rand() < args.strip_prob
    xs, ts = [], []
    brain = brain_mask(a["seg_clean_idx"])
    if getattr(args, "sr_lesion_exclude", False):   # 2026-10-29 diagnostic: drop the lesion support from the loss region.
        brain = brain & ~(a["seg_path_idx"] == LESION_IDX)
    for use_path in draws:
        lr = a["image_path"] if use_path else a["image_clean"]
        hr = a["image_path_hr"] if use_path else a["image_clean_hr"]
        if strip_fired:
          # Skull-strip augmentation (2026-10-15). Real super-resolution inputs come both ways: ATLAS, WMH and PRES are whole heads,
          # BraTS is distributed skull-stripped, and the generator renders a full head every time -- so a stripped brain is outside
          # its support. The net then over-corrects there: on ATLAS sub-551 the best global scale for its predicted residual is 1.16
          # whole-head and 0.54 on the SAME scan stripped (BraTS sits at 0.47), which is why its output scored worse than its own
          # 3 mm input on BraTS. NOT the p1-p99 frame: forcing the whole-head frame onto the stripped scan leaves the scale at 0.47,
          # and frame width alone would predict only 0.76x. It is the missing tissue class and the hard zero boundary. Strip BOTH
          # images before normalize so the draw matches deployment.
          keep = brain_mask(a["seg_path_idx"] if use_path else a["seg_clean_idx"]).float()[None, None]
          keep = F.max_pool3d(keep, 3, stride=1, padding=1)[0, 0] > 0.5   # ONE voxel of rim, deterministic: a stripping tool keeps
          lr, hr = lr * keep, hr * keep                                   # dura and a little CSF. A 0-2 voxel random dilation was
          #   tried first and rejected: on a phantom it reached into the bright extracerebral shell and moved the input PSNR by 8 dB
          #   between draws, which is a frame the net would never see at test time.
        lr_n, (lo, hi) = normalize(lr)
        if getattr(args, "sr_clean_frame", False):   # 2026-10-29 diagnostic: normalise on the CONTROL rendering's window, so the same anatomy
            _, (lo, hi) = normalize(a["image_clean"])   # is always presented at one intensity scale. Removes the frame shift a lesion causes,
            lr_n = apply_scale(lr, lo, hi)              # and leaves the unpredictable-gradient channel intact.
        hr_n = apply_scale(hr, lo, hi)
        if use_path and getattr(args, "sr_core_zero", False):   # zero-residual target inside the synthetic lesion core
            _les = (a["seg_path_idx"] == LESION_IDX)
            _core = (F.max_pool3d((~_les).float()[None, None], 5, stride=1, padding=2)[0, 0] == 0) & _les   # eroded by 2 voxels
            hr_n = torch.where(_core, lr_n, hr_n)                                                           # target = input there
        xs.append(lr_n); ts.append(hr_n)   # per-sample window, exactly as train.py normalizes each half before the cat
    x = torch.stack(xs)[:, None]         # (B,1,D,H,W): B = 1 (clean only) or 2 (clean + pathological)
    tgt = torch.stack(ts)
    with _autocast(x, args):   # 2026-10-08: --amp honoured (was hard-coded fp16)
        res = model(x)
    _logb = None
    if getattr(args, "sr_calibrate", False):
        # channel 0 = the residual, channel 1 = its amplitude through a sigmoid. Trained end to end on the
        # SAME loss below, so the amplitude is optimised against the quantity we score; no degenerate
        # solution, because shrinking it where the residual IS reliable costs error immediately.
        res = res[:, 0:1].float() * torch.sigmoid(res[:, 1:2].float())
    elif getattr(args, "sr_uncertainty", False):
        # channel 1 = log b, the per-voxel Laplace scale of the residual's error (clamped for stability). The prediction is the
        # plain residual; the loss below becomes the Laplace NLL |e|/b + log b per voxel, so the net is paid for honest b.
        _logb = res[:, 1, ...].float().clamp(-7.0, 2.0); res = res[:, 0:1].float()
    pred = (x + res.float())[:, 0]
    _err = (pred - tgt).abs()
    if _logb is not None: _err = _err * torch.exp(-_logb) + _logb
    _lw = float(getattr(args, "sr_lesion_weight", 1.0))
    if _lw != 1.0 and True in draws:   # weight the synthetic lesion support of the pathological sample
        _w = torch.ones_like(pred); _w[draws.index(True)][a["seg_path_idx"] == LESION_IDX] = _lw
        loss = (_err * _w)[:, brain].sum() / _w[:, brain].sum()
    else:
        loss = _err[:, brain].mean()   # both samples weighted equally per voxel
    _hw = float(getattr(args, "sr_lesion_hinge", 0.0))
    if _hw > 0 and True in draws:   # inside the synthetic lesion of the
        _i = draws.index(True)        # pathological sample, penalize any voxel whose output error EXCEEDS the input's (the tricubic draw the net sees);
        _les = (a["seg_path_idx"] == LESION_IDX) & brain   # zero wherever the residual improves on the input, so it only forbids getting worse
        _hinge = torch.relu((pred[_i] - tgt[_i]).abs() - (x[_i, 0] - tgt[_i]).abs())[_les]
        loss = loss + _hw * (_hinge.mean() if _hinge.numel() else pred.sum() * 0)
    lr_n, hr_n, pred = xs[0], ts[0], pred[0]     # metrics and diagnostics report the CLEAN sample (index 0); with an
    m = {"psnr_in": psnr(lr_n, hr_n, brain), "psnr_out": psnr(pred.detach(), hr_n, brain)}   # explicit use_path there is only one
    if diagnostics:
        les = (a["seg_path_idx"] == LESION_IDX) & brain
        m["psnr_lesion"] = psnr(pred.detach(), hr_n, les) if bool(les.any()) else float("nan")
        m["clip_frac"] = float(((hr < lo) | (hr > hi))[brain].float().mean())
    return loss, m


# ---------------------------------------------------------------- validation
def validate_register(model, gen, args, device, log):
    """Live pairs from the registration generator (validation_set_v2 carries the full affine augmentation, which this task does not
    train on). BN under args.bn_mode (bn_eval). 2026-10-14 diagnostics per pair, appended to the log line: mean |u| over the fixed brain
    (vox), the 12-dof least-squares affine share of u and the net translation over the fixed brain (global_field), and the lesion
    sensitivity mean |u_path - u_clean| (vox) over the moving brain -- the same draw rendered clean and pathological, no loss term."""
    d0, d1, folds, umag, share, trans, sens, uid = [], [], [], [], [], [], [], []
    with bn_eval(model, args.bn_mode), torch.no_grad():
        for _ in range(args.val_pairs):
            a = centre_on_brain(case_tensors(next(gen), device)); b = centre_on_brain(case_tensors(next(gen), device))
            while b.get("intra_pair", False):   # --intra_prob twins are training pairs; validation stays inter-subject (a fresh PAIR keeps the generator's parity)
                a = centre_on_brain(case_tensors(next(gen), device)); b = centre_on_brain(case_tensors(next(gen), device))
            a = affine_prealign(a, b, args)   # validation must sit in the same regime as training, or its gain is unreadable
            torch.set_grad_enabled(False)   # the generator yields from inside its own no_grad; keep validation grad-free either way
            mov = normalize(a["image_path"])[0]; fix = normalize(b["image_clean"])[0]
            ml, fl = a["seg_clean_idx"], b["seg_clean_idx"]
            _, flow = flow_of(model, mov, fix, args)
            labels = [l for l in torch.unique(torch.cat([ml.flatten(), fl.flatten()])).tolist() if l != 0 and l not in EXTRACEREBRAL]
            d1.append(float(label_dice_chunked(ml, fl, flow, labels)))
            d0.append(float(label_dice_chunked(ml, fl, torch.zeros_like(flow), labels)))
            folds.append(jacobian_fold_fraction(flow))
            fb, mb = brain_mask(fl), brain_mask(ml)
            umag.append(float((flow * vox_scale(flow)).norm(dim=1)[0][fb].mean()))
            s, t = global_field(flow, fb); share.append(s); trans.append(t)
            _, flow_c = flow_of(model, normalize(a["image_clean"])[0], fix, args)
            sens.append(float(((flow - flow_c) * vox_scale(flow)).norm(dim=1)[0][mb].mean()))
            mov_c = normalize(a["image_clean"])[0]                      # the moving draw registered to ITSELF: the true field is zero
            _, flow_i = flow_of(model, mov_c, mov_c, args)
            uid.append(float((flow_i * vox_scale(flow_i)).norm(dim=1)[0][mb].mean()))
    log(f"[validation] pairs={len(d1)}  dice before {np.mean(d0):.4f}  after {np.mean(d1):.4f}  "
        f"gain {np.mean(d1) - np.mean(d0):+.4f}  fold fraction {np.mean(folds):.5f}  "
        f"|u| brain {np.mean(umag):.3f} vox  affine share {np.nanmean(share):.3f}  translation {np.mean(trans):.3f} vox  "
        f"lesion sensitivity {np.mean(sens):.3f} vox  identity |u| {np.mean(uid):.3f} vox  bn {args.bn_mode}")
    return float(np.mean(d1))


def validate_sr(model, gen, args, device, log):
    """Live cases (the frozen set has no HR image). Each case is scored twice, on its clean and on its pathological rendering, so both
    arms report PSNR gain on healthy AND on pathological anatomy (2026-10-08). BN under args.bn_mode. 2026-10-14: appends the output PSNR
    inside the synthetic lesion (cases that have one) and the fraction of brain target voxels clipped by apply_scale, per rendering."""
    res = {"clean": ([], [], [], []), "path": ([], [], [], [])}
    with bn_eval(model, args.bn_mode), torch.no_grad():
        for _ in range(args.val_cases):
            a = case_tensors(next(gen), device)
            torch.set_grad_enabled(False)   # see validate_register
            for key, up in (("clean", False), ("path", True)):
                _, m = sr_step(model, a, args, use_path=up, diagnostics=True)
                res[key][0].append(m["psnr_in"]); res[key][1].append(m["psnr_out"]); res[key][2].append(m["psnr_lesion"]); res[key][3].append(m["clip_frac"])
    out = {}
    for key, (pin, pout, ples, clip) in res.items():
        out[key] = (float(np.mean(pin)), float(np.mean(pout)), float(np.nanmean(ples)) if np.isfinite(ples).any() else float("nan"), float(np.mean(clip)))
    n_les = int(np.isfinite(res["path"][2]).sum())
    log(f"[validation] cases={len(res['clean'][0])}  clean: PSNR input {out['clean'][0]:.2f} dB output {out['clean'][1]:.2f} dB gain {out['clean'][1] - out['clean'][0]:+.2f} dB  |  "
        f"pathological: input {out['path'][0]:.2f} dB output {out['path'][1]:.2f} dB gain {out['path'][1] - out['path'][0]:+.2f} dB  |  "
        f"lesion PSNR output: clean {out['clean'][2]:.2f} dB path {out['path'][2]:.2f} dB (n={n_les})  |  "
        f"target clip frac: clean {out['clean'][3]:.4f} path {out['path'][3]:.4f}  bn {args.bn_mode}")
    return out["clean"][1]


# ---------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    task = ap.add_mutually_exclusive_group(required=True)
    task.add_argument("--register", action="store_true")
    task.add_argument("--super_resolve", action="store_true")
    ap.add_argument("--sr_lesion_lowtex", type=float, default=0.0, metavar="F",
                    help="The smooth preset PLUS a low-amplitude, low-frequency interior field: texture spread at F x the "
                         "default maximum (0.1 = a tenth of the current texture contrast), texture scale >= 8 voxels (slow variation only), "
                         "and a 0.25 intra-lesion gradient, so that real lesions' slow interior gradients are in the prior without the "
                         "fine texture that the textured arms paint into real lesions. Requires --sr_smooth_lesions (applied after it).")
    ap.add_argument("--sr_smooth_lesions", action="store_true", help="2026-11-02 LESION-DEFICIT EXPERIMENT: render synthetic lesions with HOMOGENEOUS interiors. "
                    "Scored on real data, both SR arms lose to cubic interpolation INSIDE lesions while beating it brain-wide, and the pathology arm loses "
                    "0.4-0.7 dB more than the control on ATLAS/BraTS although in-domain it gains +3.95 dB inside synthetic lesions: the net has learned "
                    "to reconstruct the synthetic interior (a GMM zone up to 2x the host tissue sigma, times a +/-100%% spectral texture field at the "
                    "lesion's own spatial scale, as a core/shell ring with a sharp internal boundary) and paints that back into real lesions that are "
                    "SMOOTHER than their surround. One a-priori preset, not a sweep: texture field off, zone sigma = 0.75 x host sigma, no forced rings, "
                    "one zone per lesion. Lesion geometry, placement, contrast polarity and mass effect are untouched; the detector is not retrained.")
    ap.add_argument("--ema", type=float, default=0.0, help="Exponential moving average of the weights, decay per step (0 = off; "
                    "0.9999 = 10k-step time constant), exactly train.py's --ema of the reported detector (g500emacos). Kept in every checkpoint as "
                    "ema_state_dict (restored on --resume) and written as a sibling super_resolve_<step>_ema.pth whose model_state_dict holds the EMA "
                    "weights, so load_task_model reads it unchanged (the batch readout ignores BN running statistics). Motivation: the SR milestone "
                    "trajectories swing by more than the margins reported (ATLAS-O brain +0.87 -> +0.06 dB over cubic between 200k and 300k of one run).")
    ap.add_argument("--lr_decay_from", type=int, default=0, help="as train.py: cosine-decay the lr from this step to --steps, down to --lr_final_mult x the base lr (0 = constant)")
    ap.add_argument("--lr_final_mult", type=float, default=0.1, help="as train.py: lr multiplier reached at --steps under --lr_decay_from")
    ap.add_argument("--sr_core_zero", action="store_true", help="LESION-DEFICIT EXPERIMENT 3: inside the CORE of a synthetic lesion "
                    "(mask eroded by 2 voxels) the super-resolution target is the INPUT itself, i.e. the net is trained to apply no residual there, "
                    "while the lesion boundary (outer 2 voxels) and all healthy tissue keep the true high-resolution target. Rationale: scored on real "
                    "data the residual inside lesion cores is weakly aligned with the true correction (corr ~0.24) at ~0.6x its amplitude, so it costs "
                    "PSNR; a smooth real core carries no recoverable high-frequency content and the right action is cubic. This makes the net learn "
                    "'recognise a lesion core, leave it' end to end, the learned form of the detector gate, with the boundary still sharpened. "
                    "Combine with --sr_smooth_lesions or --sr_lesion_dr.")
    ap.add_argument("--sr_lesion_dr", action="store_true", help="LESION-DEFICIT EXPERIMENT, the domain-randomization form of "
                    "--sr_smooth_lesions: instead of fixing the interior to one smooth appearance, CENTER the randomization on smooth interiors and "
                    "keep drawing around it. Under an L1/L2 loss the net predicts the conditional mean/median over the training distribution; with "
                    "force_rings=True EVERY local/tract lesion has a sharp internal step, so that mean has a step whatever the texture draw -- a real "
                    "smooth interior lies outside the generated set, not inside its spread. This preset: rings by the primitives' own 50%% draw, "
                    "texture spread up to 0.5 with a 3-voxel floor on its spatial scale, a center->rim ramp (amplitude 0.5) so smooth gradients are "
                    "in the set, at most 2 zones, sigma cap unchanged. One a-priori preset, not a sweep.")
    ap.add_argument("--sr_lesion_hinge", type=float, default=0.0, metavar="W", help="2026-11-05: weight W on a hinge inside the synthetic lesion of the pathological sample, relu(|out-target| - |input-target|): the output may not be worse than the (tricubic) input there; 0 = off")
    ap.add_argument("--sr_lesion_weight", type=float, default=1.0, metavar="W", help="2026-11-04: weight W (e.g. 5, the detection loss's band weight) on the synthetic lesion support in the super-resolution L1, pathological sample only; 1 = the plain loss")
    ap.add_argument("--sr_lesion_exclude", action="store_true", help="2026-10-29: exclude the lesion support from the super-resolution loss region (diagnostic: is the hedging caused by unpredictable lesion gradient?)")
    ap.add_argument("--sr_clean_frame", action="store_true", help="2026-10-29: normalise the input/target on the CONTROL rendering's [p1,p99] window (diagnostic: is the hedging caused by the lesion shifting the intensity frame?)")
    ap.add_argument("--data_dir", default=os.environ.get('PR_DATA_DIR', '/path/to/data_dir'))
    ap.add_argument("--out_dir", default=os.environ.get('PR_OUT_DIR', '/path/to/out_dir'))
    ap.add_argument("--validation_dir", default=None, help="val_case_*.pt directory (registration pairs)")
    ap.add_argument("--validation_every", type=int, default=5000)
    ap.add_argument("--val_pairs", type=int, default=25)
    ap.add_argument("--val_cases", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4, help="Adam lr (constant); re-applied over the optimizer state on --resume (2026-10-14)")
    ap.add_argument("--feature_input", action="store_true", help="FeatureHeadNet: frozen --init_from trunk as the per-image feature extractor, trainable head on its d1 features (2026-10-09)")
    ap.add_argument("--init_from", default=None, metavar="CKPT",
                    help="Warm start from a trained PathoPrior / SynthSeg checkpoint (2026-10-07): the trunk loads as is (attention gates "
                         "reproduced from the checkpoint), the one-channel first conv is duplicated and halved across the two-image input, "
                         "the flow head stays zero-initialised, head/regression keys are skipped. The same weights that segment then register.")
    ap.add_argument("--resume", default=None, metavar="CKPT",
                    help="2026-10-14: continue a run from one of its own checkpoints (<run_dir>/checkpoints/<task>_NNNNN.pth saved since "
                         "today: model + optimizer + scaler + step + args + RNG). Training resumes at step N+1 up to --steps in the SAME run "
                         "directory (its train_log.txt is appended); the task and the architecture flags must equal the checkpoint's, any "
                         "other differing flag is logged as a WARNING and the new value is in force (--lr included: it is re-applied over the "
                         "checkpoint's optimizer state). Pre-2026-10-14 checkpoints (weights only) are refused: warm-start them with --init_from. "
                         "Every checkpoint saved since today is ~153 MB (weights + two Adam moments; 20 per 100k steps).")
    ap.add_argument("--attention", action="store_true",
                    help="2026-10-14: gate each skip connection on the coarser decoder feature (networks.AttentionGate), as the reported "
                         "detection model. With --init_from on a gate-less trunk the gates are the only fresh tensors; a gated checkpoint "
                         "keeps its gates whether or not this is passed.")
    ap.add_argument("--attention_dense", action="store_true",
                    help="2026-10-14: with --attention, each gate is also conditioned on every coarser decoder feature (att{1,2,3}.wg_extra.*).")
    ap.add_argument("--attention_identity_init", action="store_true",
                    help="2026-10-14: with --attention, fresh gates start fully open (psi bias 8, weight 0; sigmoid(8) = 0.9997), so a "
                         "gate-less trunk forked with --attention starts as the network it was (train.py's convention).")
    ap.add_argument("--dark_csf", type=float, default=0.0,
                    help="2026-10-14: as train.py -- probability per rendered image that the CSF GMM mean is re-drawn as WM x U[0.1,0.4] "
                         "(generator.DARK_CSF_PROB); 0 = off (the legacy prior).")
    ap.add_argument("--sr_uncertainty", action="store_true", help="Super-resolution emits a second channel, log b of a per-voxel Laplace scale on its residual error, and trains on the NLL |e|/b + log b; the prediction is the plain residual, b is the uncertainty map")
    ap.add_argument("--sr_calibrate", action="store_true", help="2026-11-07: super-resolution emits a second channel, the per-voxel amplitude of its residual through a sigmoid, trained on the same loss. Inside real lesions the residual is as large as outside while only two-thirds as well correlated with the needed correction; this lets the network scale it down there itself, from the input alone. out_channels 1 -> 2; omit the flag and the arm is unchanged.")
    ap.add_argument("--aug_rotation", type=float, default=0.0, help="2026-10-14: registration draws only -- residual affine augmentation, max rotation per axis in DEGREES (generator max_rotation), sampled independently for the moving and the fixed draw, so the pair's residual is the composition of two draws (up to twice this per axis); 0 = frame-locked pairs (today's behaviour)")
    ap.add_argument("--aug_scaling", type=float, default=0.0, help="2026-10-14: registration draws only -- max scaling per axis as a FRACTION, scale = 1 +- s (generator max_scaling), per draw as --aug_rotation; 0 = off")
    ap.add_argument("--aug_shear", type=float, default=0.0, help="2026-10-14: registration draws only -- max shear per axis as a FRACTION (generator max_shear), per draw as --aug_rotation; 0 = off")
    ap.add_argument("--bn_mode", choices=("batch", "running"), default="batch",
                    help="2026-10-14: BatchNorm convention at validation -- batch = compare_datasets.bn_batch_mode (BN layers in train mode with "
                         "momentum 0: each volume normalised by its own statistics, the detection model's readout; momentum restored after); "
                         "running = eval() running statistics (every validation before 2026-10-14). A FeatureHeadNet is unaffected (trunk "
                         "batch, head running, as before and as load_task_model).")
    ap.add_argument("--amp", choices=("fp16", "bf16", "off"), default="bf16",
                    help="fp16 + GradScaler produced a flat zero-flow registration run: the volume-wide "
                         "weight-gradient reductions overflow fp16 at the scaler's scale, the scaler skips the step, and a persistent inf skips "
                         "EVERY step while flushing the rest to zero (same signature as the tissue-sub nan runs). bf16 keeps fp32's exponent "
                         "range and needs no scaler (Ampere: A100/A6000); 'off' = fp32 everywhere (any GPU, ~2x slower).")
    ap.add_argument("--healthy", action="store_true")
    ap.add_argument("--pathology_prob", type=float, default=0.5, help="moving / input image is the pathological draw")
    ap.add_argument("--bspline_zoom", action="store_true", help="2026-10-31: upsample the simulated low-resolution acquisition back to "
                    "1 mm with a cubic B-spline (generator bspline_zooming) instead of linear. The SR task's input IS that upsample, so this "
                    "changes what the network is handed and must be matched by eval/sr_eval.py --net_input cubic. Affects registration inputs "
                    "too (shared generator); leave it off to reproduce every arm trained before 2026-10-31.")
    ap.add_argument("--lesion_comp", type=str, default=None, help="as train.py: generator.LESION_COMPOSITION, e.g. vascular:1.0,local:1.0,tract:0.5")
    ap.add_argument("--lesion_comp_f", type=str, default=None, help="as train.py: csf zone f range LO,HI")
    ap.add_argument("--comp_min_k", type=float, default=None, help="as train.py: composition visibility gate")
    ap.add_argument("--target_region", type=float, default=None, help="accepted for parity with train.py; no regression target here, so ignored")
    ap.add_argument("--lambda_smooth", type=float, default=1.0)
    ap.add_argument("--smooth_on", choices=("velocity", "displacement"), default="velocity", help="registration: field the smoothness penalty acts on (displacement = SynthMorph's form)")
    ap.add_argument("--nonlin_std", type=float, default=None, metavar="S", help="generator nonlin_std_max (default: the generator's 4). "
                    "Registration pairs differ by two independent draws of this warp, so raising it widens the shape difference the net is trained on.")
    ap.add_argument("--nonlin_scale_max", type=float, default=None, metavar="S", help="generator nonlin_scale_max (default 0.06): the control-grid "
                    "fraction. LARGER = finer grid = more local deformation; this is the knob for the 'structured vs smooth' question, not amplitude.")
    ap.add_argument("--swap_prob", type=float, default=0.0, metavar="P", help="registration: probability of swapping which draw is "
                    "moving and which is fixed, so the pathological rendering lands on the FIXED image instead (0 = never, as every "
                    "run before 2026-10-16; 0.5 covers both directions equally)")
    ap.add_argument("--consistency", type=float, default=0.0, metavar="W", help="registration: W x mean |flow(pathological moving) - flow(clean moving)| in voxels, same draw (2026-10-08)")
    ap.add_argument("--strip_prob", type=float, default=0.0, metavar="P", help="super-resolution: fraction of draws whose input AND target are skull-stripped before normalization (2026-10-15; 0 = off, as every run before it). Real inputs come both ways and the generator only ever renders a whole head")
    ap.add_argument("--affine_prealign", choices=["none", "moments", "fit"], default="none", help="registration: affinely align the moving draw onto the fixed one BEFORE the network sees the pair (SynthMorph's joint paradigm). none = the historical frame-locked pairs; moments = the closed-form centroid/scale init (free); fit = the full 12-dof label fit the evaluation uses")
    ap.add_argument("--prealign_iters", type=int, default=300, help="Adam iterations for --affine_prealign fit (the evaluation uses 300)")
    ap.add_argument("--identity_prob", type=float, default=0.0, metavar="P", help="registration: fraction of steps that ALSO register the moving draw's clean rendering to ITSELF and penalise the field (2026-10-14 degenerate-field guard; 0 = off, as every run before it)")
    ap.add_argument("--intra_prob", type=float, default=0.0, metavar="P", help="registration (2026-10-27): fraction of training pairs that are ONE subject -- the same generator draw, lesion included, under a second affine+nonlinear pose and a fresh rendering (generator intra_pair_prob); the loss is unchanged, the lesion merely appears on both sides. 0 = off (every arm before this flag)")
    ap.add_argument("--identity_weight", type=float, default=0.01, metavar="W", help="registration: W x mean |u|^2 (voxels^2) over the moving brain on those identical pairs, where the true field is zero")
    ap.add_argument("--vel_scale", type=float, default=0.1, help="scale on the raw network output before integration")
    ap.add_argument("--int_steps", type=int, default=7)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gen_device", default=None)
    ap.add_argument("--prefetch", action="store_true")
    return ap.parse_args(argv)


def main():
    args = parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device)
    gen_device = torch.device(args.gen_device) if args.gen_device else device
    pathology_primitives.DISABLED_MODES = {"Global Pathology", "Geometric Atrophy"}
    import generator as _gen
    if getattr(args, "sr_smooth_lesions", False):   # 2026-11-02: homogeneous lesion interiors (see --help); module globals read at draw time
        _gen.LESION_TEXTURE_SPREAD_MAX = 0.0; _gen.LESION_SIGMA_FRACTION = 0.75; _gen.LESION_FORCE_RINGS = False; _gen.LESION_MAX_ZONES = 1
        print(f"[sr_smooth_lesions] texture_spread_max={_gen.LESION_TEXTURE_SPREAD_MAX} sigma_fraction={_gen.LESION_SIGMA_FRACTION} "
              f"force_rings={_gen.LESION_FORCE_RINGS} max_zones={_gen.LESION_MAX_ZONES}", flush=True)
    if getattr(args, "sr_lesion_lowtex", 0.0) > 0:   # 2026-11-04: smooth + low-amplitude low-frequency interior (see --help)
        if not getattr(args, "sr_smooth_lesions", False): raise SystemExit("--sr_lesion_lowtex requires --sr_smooth_lesions")
        _gen.LESION_TEXTURE_SPREAD_MAX = float(args.sr_lesion_lowtex) * 1.0; _gen.LESION_TEXTURE_MIN_SCALE = 8.0; _gen.INTRA_LESION_GRADIENT = 0.25
        print(f"[sr_lesion_lowtex] texture_spread_max={_gen.LESION_TEXTURE_SPREAD_MAX} texture_min_scale={_gen.LESION_TEXTURE_MIN_SCALE} "
              f"intra_lesion_gradient={_gen.INTRA_LESION_GRADIENT} (sigma_fraction/rings/zones as sr_smooth_lesions)", flush=True)
    if getattr(args, "sr_lesion_dr", False):   # smooth-centered randomization (see --help)
        if getattr(args, "sr_smooth_lesions", False): raise SystemExit("--sr_lesion_dr and --sr_smooth_lesions are alternatives")
        _gen.LESION_FORCE_RINGS = False; _gen.LESION_TEXTURE_SPREAD_MAX = 0.5; _gen.LESION_TEXTURE_MIN_SCALE = 3.0
        _gen.INTRA_LESION_GRADIENT = 0.5; _gen.LESION_MAX_ZONES = 2
        print(f"[sr_lesion_dr] force_rings={_gen.LESION_FORCE_RINGS} texture_spread_max={_gen.LESION_TEXTURE_SPREAD_MAX} "
              f"texture_min_scale={_gen.LESION_TEXTURE_MIN_SCALE} intra_lesion_gradient={_gen.INTRA_LESION_GRADIENT} max_zones={_gen.LESION_MAX_ZONES}", flush=True)
    if args.lesion_comp:
        _gen.LESION_COMPOSITION = {k: float(v) for k, v in (kv.split(":") for kv in args.lesion_comp.split(","))}
    if args.lesion_comp_f:
        _gen.COMP_F_CSF = tuple(float(v) for v in args.lesion_comp_f.split(","))
    if args.comp_min_k is not None:
        _gen.COMP_MIN_K = float(args.comp_min_k)

    task_name = "register" if args.register else "super_resolve"
    ck = None
    if args.resume:
        # --resume (2026-10-14): same run directory as the checkpoint (<run_dir>/checkpoints/<ckpt>), its log appended
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        check_resume(ck, args, args.resume)
        # generator / loss flags are not refused, but a change is the user's to notice: list every differing arg that is not run plumbing
        changed = {k: (ck["args"][k], v) for k, v in vars(args).items() if k in ck["args"] and ck["args"][k] != v and k not in
                   ("resume", "steps", "init_from", "device", "gen_device", "prefetch", "validation_every", "val_pairs", "val_cases", "validation_dir", "out_dir", "data_dir", "tag")}
        ckpt_dir = os.path.dirname(os.path.abspath(args.resume))
        if os.path.basename(ckpt_dir) != "checkpoints":
            raise SystemExit(f"--resume expects <run_dir>/checkpoints/<ckpt>.pth, got {args.resume}")
        run_dir = os.path.dirname(ckpt_dir)
    else:
        prefix = "_".join([task_name] + (["healthy"] if args.healthy else []) + ([args.tag] if args.tag else []) + [git_id()])
        run_dir = os.path.join(args.out_dir, f"{prefix}_out", f"log_{datetime.now():%Y%m%d_%H%M%S}")
        ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, "train_log.txt"), "a")

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    log(f"{task_name} -> {run_dir}")
    for k, v in sorted(vars(args).items()):
        log(f"    {k:20s} {v}")

    in_ch, out_ch = task_channels(args)
    if ck is not None and "feature_input_flag" in ck["model_state_dict"]:
        model = feature_head_from_state(ck["model_state_dict"], out_ch, device)      # resumed feature-input run: trunk inside the checkpoint
    elif args.feature_input:
        if not args.init_from: raise SystemExit("--feature_input needs --init_from <detection checkpoint> (the frozen trunk)")
        from networks import load_checkpoint
        trunk, _, _ = load_checkpoint(args.init_from, device, verbose=False)
        model = FeatureHeadNet(trunk, in_ch, out_ch).to(device)
        log(f"feature_input: frozen trunk {args.init_from} ({model.c}-ch d1 features x {in_ch} images) -> trainable head; trunk BN on batch statistics")
    elif ck is not None:
        model, _ = build_model(args, None, device, log, arch_state=ck["model_state_dict"])   # architecture from the run's own checkpoint
        if args.init_from:
            log(f"--init_from {args.init_from} ignored under --resume (weights come from the resumed checkpoint)")
    elif args.init_from:
        ck_i = torch.load(args.init_from, map_location="cpu", weights_only=False)
        init_state = ck_i["model_state_dict"] if (isinstance(ck_i, dict) and "model_state_dict" in ck_i) else ck_i
        log(f"init_from {args.init_from}")
        model, _ = build_model(args, init_state, device, log)
    else:
        model, _ = build_model(args, None, device, log)
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=args.lr)
    _ema = {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()} if args.ema > 0 else None
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))   # only fp16 needs loss scaling (see --amp)
    log(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    log(f"bn_mode {args.bn_mode}: validation BN on " + ("per-volume batch statistics (compare_datasets.bn_batch_mode)" if args.bn_mode == "batch" else "running statistics"))
    start_step = 0
    if ck is not None:
        # RNG restored before the generator is built, so its first draws follow the saved streams (with --prefetch the producer thread
        # and the training loop interleave their draws, so the sequence is not bit-reproducible there either way)
        start_step = load_resume(ck, model, opt, scaler, log)
        for g in opt.param_groups:
            g["lr"] = args.lr          # opt.load_state_dict restores the checkpoint's lr; --lr of THIS command is the one in force (2026-10-14)
        if _ema is not None and isinstance(ck, dict) and "ema_state_dict" in ck:
            for k, v in ck["ema_state_dict"].items():
                if k in _ema: _ema[k].copy_(v.to(_ema[k].device))
            log(f"ema {args.ema}: ema_state_dict restored ({len(_ema)} tensors)")
        elif _ema is not None: log(f"ema {args.ema}: checkpoint has no ema_state_dict -- EMA re-initialised from the resumed weights")
        log(f"resumed from {args.resume} at step {start_step} (code {git_id()})")
        if changed:
            log(f"WARNING: flags differ from the checkpoint's run (checkpoint, now): {changed}")
        if start_step >= args.steps:
            log(f"nothing to do: checkpoint step {start_step} >= --steps {args.steps}"); log("TRAINING COMPLETE"); return
    if args.super_resolve and (args.aug_rotation or args.aug_scaling or args.aug_shear):
        log("WARNING: --aug_* apply to registration draws only; SR keeps the generator's default affine augmentation")

    set_dark_csf(args.dark_csf, log)                     # before the generator / prefetcher (module global, thread-shared)
    gen = build_generator(args, gen_device)
    if args.prefetch:
        gen = Prefetcher(gen, depth=2)

    roll = {}
    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        a = case_tensors(next(gen), device)
        if args.register:
            b = case_tensors(next(gen), device)
        torch.set_grad_enabled(True)   # the generator yields from inside no_grad; without --prefetch that would leak into this step (2026-10-07; moved out of the step functions 2026-10-14 so validation stays grad-free)
        if args.register:
            loss, m = register_step(model, a, b, args)
        else:
            loss, m = sr_step(model, a, args)
        w0 = model.final_conv.weight.detach().clone()
        if args.lr_decay_from > 0 and step >= args.lr_decay_from:   # train.py's cosine, base lr = --lr
            _frac = min(1.0, (step - args.lr_decay_from) / max(1, args.steps - args.lr_decay_from))
            _m = args.lr_final_mult + (1.0 - args.lr_final_mult) * 0.5 * (1.0 + math.cos(math.pi * _frac))
            for g in opt.param_groups: g["lr"] = args.lr * _m
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if _ema is not None:
            with torch.no_grad():
                for _k, _v in model.state_dict().items():
                    if _k in _ema: _ema[_k].mul_(args.ema).add_(_v.detach(), alpha=1.0 - args.ema)
        m["skip"] = float(torch.equal(w0, model.final_conv.weight.detach()))   # 1 = optimizer step skipped (non-finite grads) or weights frozen: a flat run shows here within 100 steps
        m["scale"] = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
        for k, v in m.items():
            roll[k] = 0.98 * roll.get(k, v) + 0.02 * v
        if step % 100 == 0:
            log(f"Step {step:06d} | loss {float(loss):.4f} | " + " ".join(f"{k} {v:.4f}" for k, v in roll.items()) +
                f" | {(time.time() - t0) / (step - start_step):.2f}s/step")
        if step % 5000 == 0:
            p = os.path.join(ckpt_dir, f"{task_name}_{step:05d}.pth")
            save_checkpoint(p, model, opt, scaler, step, args, ema=_ema)
            log(f"Saved Checkpoint: {p}")
            if _ema is not None:   # sibling with the EMA weights in model_state_dict, loadable by load_task_model like any checkpoint
                pe = p.replace(".pth", "_ema.pth"); torch.save({"model_state_dict": {**model.state_dict(), **_ema}, "step": int(step), "ema_decay": args.ema}, pe); log(f"Saved EMA weights: {pe}")
        if args.validation_every and step % args.validation_every == 0:
            if args.register:
                validate_register(model, gen, args, device, log)
            elif args.super_resolve:
                validate_sr(model, gen, args, device, log)
    log("TRAINING COMPLETE")


if __name__ == "__main__":
    main()
