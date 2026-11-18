"""Shared network definitions.

SynthSegUNet is defined once here and imported everywhere it's needed, so a
single architecture change applies everywhere it's used.

The norm layer defaults to BatchNorm3d, matching SynthSeg (batch_norm=-1).
It stays a constructor argument because when loading a checkpoint the choice
is not free -- the state_dict determines it. BatchNorm3d stores
running_mean/running_var/num_batches_tracked; InstanceNorm3d (with
track_running_stats=False) does not, so loading the wrong one fails on
missing or unexpected keys. Use detect_norm() when loading an existing
checkpoint rather than assuming.

Regression head variants, both off by default so every existing checkpoint
loads unchanged: reg_head='mlp' replaces the single 1x1 conv by
conv-ELU-conv (keys regression_mlp.0/2.*; 'linear' keeps regression_conv.*),
and tissue_rel=True appends TissueRelative's per-scan tissue-relative
intensity channel to the head input (marked in the state_dict by the
tissue_rel_flag buffer). detect_reg_head() / detect_tissue_rel() read both
off a state_dict; load_checkpoint_into() loads an old head-less or
linear-head checkpoint into a model with a new head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch and Keras define BatchNorm momentum inversely:
#   Keras:   moving  = momentum * moving  + (1 - momentum) * batch   (default 0.99)
#   PyTorch: running = (1 - momentum) * running + momentum * batch   (default 0.10)
# so PyTorch momentum = 1 - Keras momentum. SynthSeg uses the Keras default,
# an effective PyTorch momentum of 0.01. PyTorch's own 0.1 default gives a
# 10x shorter averaging window -- roughly the last 10 batches -- which is
# too short when contrast is re-randomized every step at batch size 1.
BN_MOMENTUM = 0.01


def batchnorm(c):
    """BatchNorm3d with the SynthSeg-equivalent momentum."""
    return nn.BatchNorm3d(c, momentum=BN_MOMENTUM)


class PerSampleBatchNorm3d(nn.BatchNorm3d):
    """BatchNorm3d whose TRAINING statistics are per sample: each element of the batch is
    normalized with its own channel mean/var (instance statistics, BatchNorm affine), so the
    (clean, path) pair is never pooled and training matches the per-scan readout exactly.
    eval() is unchanged (running buffers); the buffers track the sample-averaged statistics."""
    def forward(self, x):
        if not self.training:
            return super().forward(x)
        n, c = x.shape[:2]
        if self.track_running_stats and self.momentum:
            with torch.no_grad():
                m = x.float().mean(dim=(0, 2, 3, 4)); v = x.float().var(dim=(2, 3, 4), unbiased=True).mean(0)
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * m)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * v)
                self.num_batches_tracked += 1
        y = F.batch_norm(x.reshape(1, n * c, *x.shape[2:]), None, None,
                         self.weight.repeat(n), self.bias.repeat(n), True, 0.0, self.eps)
        return y.reshape(x.shape)


def per_sample_bn(model):
    """Swap every BatchNorm3d for PerSampleBatchNorm3d in place (same parameters, buffers and
    state_dict keys, so checkpoints load either way). Returns the count."""
    n = 0
    for m in model.modules():
        if type(m) is nn.BatchNorm3d:
            m.__class__ = PerSampleBatchNorm3d; n += 1
    return n


def detect_norm(state_dict):
    """Return the norm layer a checkpoint was trained with.

    Detect rather than assume: checkpoints predating the BatchNorm switch are
    InstanceNorm, and constructing with the default (BatchNorm) would fail on
    the missing running_mean/running_var/num_batches_tracked buffers.
    """
    return batchnorm if any(k.endswith("running_mean") for k in state_dict) else nn.InstanceNorm3d


def detect_regression_channels(state_dict):
    """Return the regression head's channel count a checkpoint was trained
    with (0 if it has none).

    Same reasoning as detect_norm(): a checkpoint predating the regression
    head has no regression_conv.* keys at all, and constructing with a
    nonzero default would create a parameter the state_dict doesn't have,
    failing load_state_dict on an unexpected-missing-key mismatch. Reading
    the channel count off the weight's own shape (rather than hardcoding an
    expected value) means this keeps working if that count ever changes.
    """
    keys = _regression_weight_keys(state_dict)
    return state_dict[keys[1]].shape[0] if keys else 0


REGRESSION_HEAD_PREFIXES = ("regression_conv.", "regression_mlp.")
TISSUE_REL_FLAG = "tissue_rel_flag"
TISSUE_POOL_FLAG = "tissue_pool_flag"   # 2026-09-29: which pools the own-class channel uses (TissueRelative.POOLS code); absent = 'own'
TISSUE_SOFT_FLAG = "tissue_soft_flag"   # 2026-10-07: own-class channel is the class-POSTERIOR-weighted mixture of per-class z (marker only)
TISSUE_UNROLL_FLAG = "tissue_unroll_flag"   # 2026-09-29: number of re-estimation passes (pools recomputed without the head's outliers); absent = 0


def is_regression_head_key(key):
    """True for the regression head's own state_dict keys (and the tissue_rel
    marker buffer), i.e. everything load_checkpoint_into may re-initialise."""
    return key.startswith(REGRESSION_HEAD_PREFIXES) or key == TISSUE_REL_FLAG


def _regression_weight_keys(state_dict):
    """(first, last) layer weight keys of the regression head, or None."""
    mlp = sorted((k for k in state_dict if k.startswith("regression_mlp.") and k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
    if mlp:
        return mlp[0], mlp[-1]
    if "regression_conv.weight" in state_dict:
        return "regression_conv.weight", "regression_conv.weight"
    return None


def detect_reg_arch(state_dict):
    """(hidden, depth, kernel) of an MLP regression head: hidden width, number of conv layers, kernel of the first layer.
    (32, 2, 1) is the original conv-ELU-conv head; a linear head reports (0, 1, 1)."""
    mlp = sorted((k for k in state_dict if k.startswith("regression_mlp.") and k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
    if not mlp:
        return 0, 1, 1
    w0 = state_dict[mlp[0]]
    return int(w0.shape[0]), len(mlp), int(w0.shape[2])


def detect_tissue_unroll(state_dict):
    """Number of pool re-estimation passes the head was trained with (marker buffer; absent = 0)."""
    return int(state_dict[TISSUE_UNROLL_FLAG]) if TISSUE_UNROLL_FLAG in state_dict else 0


def detect_reg_head(state_dict):
    """'mlp' if the checkpoint's regression head is conv-ELU-conv, else 'linear'."""
    return "mlp" if "regression_mlp.0.weight" in state_dict else "linear"


def detect_tissue_rel(state_dict):
    """True if the regression head reads the tissue-relative channel: the
    model registers the tissue_rel_flag buffer only when it does."""
    return TISSUE_REL_FLAG in state_dict and int(state_dict[TISSUE_REL_FLAG]) != 0


# tissue_rel_flag buffer value = channel count of TissueRelative's output = 1 (own class) + refs + vec channels.
# The layouts are chosen so every count maps to exactly one (refs, vec) pair.
_TR_LAYOUT = {1: (0, "none"), 4: (3, "none"), 3: (0, "minmax"), 6: (3, "minmax"), 12: (0, "full"), 15: (3, "full")}


def detect_tissue_refs(state_dict):
    """Number of fixed-reference tissue channels (0 for the single per-class channel)."""
    return _TR_LAYOUT[int(state_dict[TISSUE_REL_FLAG])][0] if TISSUE_REL_FLAG in state_dict else 0


def detect_tissue_pool(state_dict):
    """'own' | 'lr' | 'mirror' | 'coarse' | 'coarse_th': the pools the own-class channel uses (marker buffer; absent = 'own')."""
    code = int(state_dict[TISSUE_POOL_FLAG]) if TISSUE_POOL_FLAG in state_dict else 0
    return TissueRelative.POOLS[code]


def detect_tissue_soft(state_dict):
    """True when the own-class channel was trained as the posterior-weighted mixture (marker buffer, 2026-10-07)."""
    return TISSUE_SOFT_FLAG in state_dict


def detect_tissue_vec(state_dict):
    """'none' | 'minmax' | 'full': the class-vector channels the regression head reads (2026-09-29)."""
    return _TR_LAYOUT[int(state_dict[TISSUE_REL_FLAG])][1] if TISSUE_REL_FLAG in state_dict else "none"


def head_description(reg_head, tissue_rel, tissue_refs=0, tissue_vec="none", tissue_pool="own", tissue_soft=False):
    """'linear' | 'mlp' | 'linear+tissue_rel' | 'mlp+tissue_rel' | 'mlp+tissue_rel+refs3' | 'mlp+tissue_rel+vecminmax' | '...+pool:coarse'."""
    return (reg_head + ("+tissue_rel" if tissue_rel else "") + (f"+pool:{tissue_pool}" if tissue_rel and tissue_pool != "own" else "")
            + ("+soft" if tissue_rel and tissue_soft else "")
            + (f"+refs{tissue_refs}" if tissue_rel and tissue_refs else "") + (f"+vec{tissue_vec}" if tissue_rel and tissue_vec != "none" else ""))


def detect_in_channels(state_dict):
    """Return the input channel count a checkpoint was trained with.

    Registration takes a moving/fixed pair, so it is the first task with more
    than one input channel; without this a 2-channel checkpoint cannot be
    reconstructed from its state_dict alone.
    """
    key = "enc1.conv.0.weight"
    return state_dict[key].shape[1] if key in state_dict else 1


def detect_attention(state_dict):
    """True if the checkpoint was trained with skip attention gates."""
    return any(k.startswith("att4.") for k in state_dict)


def detect_attention_dense(state_dict):
    """True if the gates take extra coarser-decoder gating signals (--attention_dense)."""
    return any(k.startswith("att1.wg_extra.") for k in state_dict)


def detect_seg_prior(state_dict):
    """True if the regression head reads the segmentation softmax as well as
    the decoder features: its input width is then the decoder width plus the
    number of classes, both of which final_conv's weight already records."""
    keys = _regression_weight_keys(state_dict)
    if keys is None:
        return False
    fc = state_dict["final_conv.weight"]
    extra = state_dict[keys[0]].shape[1] - fc.shape[1] - (int(state_dict[TISSUE_REL_FLAG]) if detect_tissue_rel(state_dict) else 0)
    return extra == fc.shape[0]


def detect_probability_channels(state_dict):
    """Return the probability head's channel count a checkpoint was trained
    with (0 if it has none). Same reasoning as detect_regression_channels()."""
    key = "probability_conv.weight"
    return state_dict[key].shape[0] if key in state_dict else 0


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, norm=batchnorm):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            norm(out_c),
            nn.ELU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            norm(out_c),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """Lets the coarse decoder feature decide which skip features pass.

    g and x are at the same resolution, so no strided projection is needed:
    the gating signal is the already-upsampled decoder feature.

    extra_gate_c (--attention_dense) adds the decoder features from every
    COARSER level as further gating signals, so the gate conditions its
    decision on several scales instead of one. This is not a receptive-field
    or a capacity change -- the measured effective RF is already ~132 of 140
    voxels, and the regression head reads nothing new. It changes WHICH
    representation controls the gate: a lesion's evidence may sit at a
    different scale than the single adjacent level that currently gates it.
    Each extra signal is projected at ITS OWN resolution and only then
    upsampled (projecting after upsampling would put a 384-channel tensor at
    160^3, ~6 GB for one sample), and it is SUMMED into the same
    pre-activation, so inter_c and psi are unchanged and no new width
    hyperparameter appears.
    """

    def __init__(self, gate_c, skip_c, inter_c=None, identity_init=False, extra_gate_c=()):
        super().__init__()
        inter_c = inter_c or max(skip_c // 2, 1)
        self.wg = nn.Conv3d(gate_c, inter_c, kernel_size=1)
        self.wx = nn.Conv3d(skip_c, inter_c, kernel_size=1)
        self.wg_extra = nn.ModuleList([nn.Conv3d(int(c), inter_c, kernel_size=1) for c in extra_gate_c])
        self.psi = nn.Conv3d(inter_c, 1, kernel_size=1)
        if identity_init:
            # gate opens fully at init (sigmoid(8) = 0.9997): a gate-less checkpoint forked with --attention starts
            # as the network it was, and the gates learn where to close. Zero psi weight => no input dependence at init.
            # Holds for the dense variant too: psi is the only path to the gate, so extra signals start inert.
            nn.init.zeros_(self.psi.weight); nn.init.constant_(self.psi.bias, 8.0)

    def forward(self, g, x, extra=()):
        h = self.wg(g) + self.wx(x)
        for conv, e in zip(self.wg_extra, extra):
            h = h + F.interpolate(conv(e), size=x.shape[2:], mode="trilinear", align_corners=False)
        return x * torch.sigmoid(self.psi(F.relu(h)))


_SUBSAMPLE_GENERATORS = {}


def _subsample_generator(device):
    """Private RNG stream per device, so the median/MAD subsample never
    touches the training stream."""
    key = str(device)
    if key not in _SUBSAMPLE_GENERATORS:
        g = torch.Generator(device=device)
        g.manual_seed(0)
        _SUBSAMPLE_GENERATORS[key] = g
    return _SUBSAMPLE_GENERATORS[key]


class TissueRelative(nn.Module):
    """Per-scan tissue-relative intensity, no parameters: for each batch
    element and anatomy class c (argmax of the DETACHED segmentation logits)
    rel = clamp((x - median_c) / (1.4826 MAD_c + eps), -clamp, clamp), the
    statistics over that class's own voxels in the same image. A
    contrast-agnostic net can read the class off the input but has no fixed
    class -> intensity map; this hands the head the per-scan comparison
    (eval/tissue_inconsistency.py's rule) as one input channel.

    Compact indices of LABEL_LIST_SEGMENTATION (train.py): 7 = Lesion,
    scored against white matter (20/38 pooled, lesion voxels excluded from
    its statistics); 0, the extracerebral block 8-19 and any class with
    fewer than min_voxels reference voxels get rel = 0. Median/MAD come
    from torch.quantile on at most max_sample voxels per class.
    """
    LESION = 7
    WM = (20, 38)
    SKIP = (0,) + tuple(range(8, 20))
    REFS = ((20, 38), (21, 39), (4, 22, 40))   # fixed references: white matter, cortex, CSF (sulcal + lateral ventricles)
    # class vector (2026-09-29): every brain voxel scored against each of these healthy pools
    #   vec='minmax': 2 channels, min_c |z_c| (intensity anomaly against ALL classes, no segmentation of the voxel needed)
    #                 and the gap between the two smallest |z_c| (small = intensity between two classes); both clamped to [0, clamp]
    #   vec='full':   one signed channel per reference, clamped to +-clamp
    REFS_ALL = ((20, 38), (21, 39), (4,), (22, 40), (26, 44), (27, 45), (28, 46), (29, 47), (3,), (24, 42), (25, 43))
    VEC_CHANNELS = {"none": 0, "minmax": 2, "full": len(REFS_ALL)}
    # pool (2026-09-29): which voxels form the pool the own-class channel scores a voxel against. Lesion voxels (7) are always
    # scored vs (the pool that contains) WM and excluded from every pool; extracerebral / background / spinal cord stay skipped.
    #   own        its own compact class (L/R cerebral WM pooled), the original behaviour
    #   lr         its class with the homologous contralateral class pooled in (halves the share a unilateral lesion can take)
    #   mirror     the CONTRALATERAL homologous class only (a unilateral lesion never contaminates its own reference); midline classes: own
    #   coarse     its tissue type: 1 WM, 2 GM, 3 CSF (a lesion is a small fraction of a large pool)
    #   coarse_th  coarse with the thalamus as a 4th type (its intensity sits between WM and GM on most contrasts)
    POOLS = ("own", "lr", "mirror", "coarse", "coarse_th")
    LR_PAIRS = tuple((l, l + 18) for l in range(20, 38)) + ((56, 57),)   # (20,38) WM ... (37,55) septal nuclei, (56,57) choroid plexus (--choroid_target); midline 1-7 unpaired
    COARSE_WM = (3, 5, 6, 7, 20, 24, 34, 38, 42, 52)
    COARSE_GM = (21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 39, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57)   # choroid plexus: vascular tissue, pooled with GM
    COARSE_CSF = (1, 2, 4, 22, 23, 40, 41)
    THALAMUS = (26, 44)

    def __init__(self, min_voxels=200, max_sample=200_000, clamp=10.0, eps=1e-3, refs=0, vec="none", pool="own", soft=False):
        super().__init__()
        self.min_voxels, self.max_sample, self.clamp, self.eps = min_voxels, max_sample, clamp, eps
        if pool not in self.POOLS:
            raise ValueError(f"pool must be one of {self.POOLS}, got {pool!r}")
        self.pool = pool
        # soft own-class channel (2026-10-07): rel = sum_c p_c z_c with p = softmax(logits) (detached) and z_c the voxel's
        # deviation under class c's statistics, instead of z under the argmax class alone. At a lesion boundary the argmax flips
        # to cortex and the hard channel compares the voxel with cortex, where it looks normal (leg_failure_signature: 37-51 %
        # of the missed ATLAS / BraTS-T2w voxels sit under the cortex class); the mixture keeps the WM comparison alive in
        # proportion to the posterior. Pools other than 'own' are not combined with it.
        if soft and pool != "own":
            raise ValueError("soft own-class channel needs pool='own'")
        self.soft = soft
        # lookup tables (compact id -> pool id) for the own-class channel; 0 = skip
        lr = torch.arange(64); mirror = torch.arange(64)   # sized past the 58-class list (--choroid_target)
        for l, r in self.LR_PAIRS:
            lr[r] = l; mirror[l], mirror[r] = r, l
        coarse = torch.zeros(64, dtype=torch.long)
        for group, ids in ((1, self.COARSE_WM), (2, self.COARSE_GM), (3, self.COARSE_CSF)):
            coarse[list(ids)] = group
        coarse_th = coarse.clone(); coarse_th[list(self.THALAMUS)] = 4
        self.register_buffer("_lut_lr", lr, persistent=False); self.register_buffer("_lut_mirror", mirror, persistent=False)
        self.register_buffer("_lut_coarse", coarse, persistent=False); self.register_buffer("_lut_coarse_th", coarse_th, persistent=False)
        self.refs = self.REFS[:refs]   # refs=3: three more channels, every brain voxel against each reference tissue
        if vec not in self.VEC_CHANNELS:
            raise ValueError(f"vec must be one of {list(self.VEC_CHANNELS)}, got {vec!r}")
        self.vec = vec
        self.n_vec = self.VEC_CHANNELS[vec]

    @torch.no_grad()
    def forward(self, x, logits, exclude=None):
        """x (B, C, D, H, W): channel 0 is the image; logits (B, K, D, H, W); exclude (B, D, H, W) bool, optional:
        voxels removed from every pool (still scored) -- the unrolled re-estimation passes the head's outliers here.
        Returns (B, 1 + len(refs) + n_vec, D, H, W) in x's dtype."""
        img = x[:, 0].float()
        cls = logits.detach().argmax(dim=1)
        excl = exclude if exclude is not None else torch.zeros_like(cls, dtype=torch.bool)
        rel = torch.zeros((img.shape[0], 1 + len(self.refs) + self.n_vec) + img.shape[1:], device=img.device, dtype=img.dtype)
        for b in range(img.shape[0]):
            xb, cb, eb = img[b], cls[b], excl[b]
            wm = torch.full_like(cb, self.WM[0])
            scored = torch.where((cb == self.LESION) | (cb == self.WM[1]), wm, cb)
            ref = torch.where((cb == self.LESION) | eb, torch.full_like(cb, -1), scored)      # fine pools: refs / vec channels use these
            if self.pool in ("coarse", "coarse_th"):
                lut = (self._lut_coarse if self.pool == "coarse" else self._lut_coarse_th).to(cb.device)
                own_scored = lut[cb]                                                    # lesion (7) -> WM group
                own_ref = torch.where((cb == self.LESION) | eb, torch.full_like(cb, -1), own_scored); own_skip = (0,)
                for k in torch.unique(own_scored).tolist():
                    if k in own_skip:
                        continue
                    med, mad = self._stats(xb[own_ref == k])
                    if med is None:
                        continue
                    m = own_scored == k
                    rel[b, 0][m] = ((xb[m] - med) / (mad + self.eps)).clamp(-self.clamp, self.clamp)
            elif self.pool in ("lr", "mirror"):
                # score group = the voxel's own class (lesion -> left WM); reference pool per group:
                #   lr: the class and its homologue pooled;  mirror: the homologue only (own class when it has no homologue)
                own_scored = torch.where(cb == self.LESION, torch.full_like(cb, self.WM[0]), cb)
                ref_cls = torch.where((cb == self.LESION) | eb, torch.full_like(cb, -1), cb)
                lut_lr, lut_mirror = self._lut_lr.to(cb.device), self._lut_mirror.to(cb.device)
                for k in torch.unique(own_scored).tolist():
                    if k in self.SKIP:
                        continue
                    if self.pool == "lr":
                        pool_mask = lut_lr[ref_cls.clamp(min=0)] == lut_lr[k]
                    else:
                        pool_mask = ref_cls == int(lut_mirror[k])
                    pool_mask = pool_mask & (ref_cls >= 0)
                    med, mad = self._stats(xb[pool_mask])
                    if med is None:
                        continue
                    m = own_scored == k
                    rel[b, 0][m] = ((xb[m] - med) / (mad + self.eps)).clamp(-self.clamp, self.clamp)
            elif self.soft:
                lb = logits[b].detach().float()
                lse = torch.logsumexp(lb, dim=0)
                brain = ~torch.isin(cb, torch.tensor(self.SKIP, device=cb.device))
                acc = torch.zeros_like(xb); wsum = torch.zeros_like(xb)
                for k in range(lb.shape[0]):
                    if k in self.SKIP:
                        continue
                    kk = self.WM[0] if k in (self.LESION, self.WM[1]) else k   # lesion and right WM share the pooled WM statistics
                    med, mad = self._stats(xb[ref == kk])
                    if med is None:
                        continue
                    p = torch.exp(lb[k] - lse)
                    acc += p * ((xb - med) / (mad + self.eps)).clamp(-self.clamp, self.clamp)
                    wsum += p
                ok = brain & (wsum > 1e-3)
                rel[b, 0][ok] = acc[ok] / wsum[ok]
            else:
                for k in torch.unique(scored).tolist():
                    if k in self.SKIP:
                        continue
                    med, mad = self._stats(xb[ref == k])
                    if med is None:
                        continue
                    m = scored == k
                    rel[b, 0][m] = ((xb[m] - med) / (mad + self.eps)).clamp(-self.clamp, self.clamp)
            if self.refs:
                brain = ~torch.isin(cb, torch.tensor(self.SKIP, device=cb.device))
                for j, ids in enumerate(self.refs):
                    med, mad = self._stats(xb[(ref.unsqueeze(0) == torch.tensor(ids, device=cb.device).view(-1, 1, 1, 1)).any(0)])
                    if med is None:
                        continue
                    rel[b, 1 + j][brain] = ((xb[brain] - med) / (mad + self.eps)).clamp(-self.clamp, self.clamp)
            if self.n_vec:
                brain = ~torch.isin(cb, torch.tensor(self.SKIP, device=cb.device)); base = 1 + len(self.refs)
                zs = []
                for ids in self.REFS_ALL:
                    med, mad = self._stats(xb[torch.isin(ref, torch.tensor(ids, device=cb.device))])
                    zs.append(None if med is None else (xb - med) / (mad + self.eps))
                if self.vec == "full":
                    for j, zc in enumerate(zs):
                        if zc is not None:
                            rel[b, base + j][brain] = zc[brain].clamp(-self.clamp, self.clamp)
                else:
                    az = [zc.abs() for zc in zs if zc is not None]
                    if az:
                        A = torch.stack(az, 0)
                        if A.shape[0] >= 2:
                            two = torch.topk(A, 2, dim=0, largest=False).values     # the two smallest |z_c| per voxel
                            zmin, gap = two[0], two[1] - two[0]
                        else:
                            zmin, gap = A[0], torch.zeros_like(A[0])
                        rel[b, base][brain] = zmin[brain].clamp(0, self.clamp)
                        rel[b, base + 1][brain] = gap[brain].clamp(0, self.clamp)
        return rel.to(x.dtype)

    def _stats(self, v):
        """Median and 1.4826 MAD of a reference sample, or (None, None) below min_voxels."""
        if v.numel() < self.min_voxels:
            return None, None
        if v.numel() > self.max_sample:
            idx = torch.randint(v.numel(), (self.max_sample,), device=v.device, generator=_subsample_generator(v.device))
            v = v[idx]
        med = torch.quantile(v, 0.5)
        return med, 1.4826 * torch.quantile((v - med).abs(), 0.5)


class SynthSegUNet(nn.Module):
    """5-level UNet matching SynthSeg's architecture.

    unet_feat_count=24, feat_mult=2, nb_conv_per_level=2, activation=elu.
    SynthSeg uses batch_norm=-1 (BatchNormalization on the channel axis), which
    is the default here. Pass norm=nn.InstanceNorm3d for pre-switch checkpoints,
    or let detect_norm/load_checkpoint pick it from the state_dict.

    regression_channels: size of a second 1x1-conv head branching off the
    same final decoder features as the segmentation head, for a continuous
    target trained alongside segmentation. 0 (default) omits the head
    entirely, so checkpoints without one load unchanged -- see
    detect_regression_channels().

    reg_head: 'linear' (the 1x1 conv, regression_conv.*) or 'mlp'
    (conv-ELU-conv, 32 hidden, regression_mlp.*). tissue_rel: the head also
    reads TissueRelative(x, logits), one channel. Both leave the default
    parameter set and forward untouched.
    """

    def __init__(self, in_channels=1, out_channels=56, init_filters=24,
                 norm=batchnorm, regression_channels=0, probability_channels=0,
                 attention=False, seg_prior=False, reg_head="linear", tissue_rel=False, tissue_refs=0,
                 attention_identity_init=False, tissue_vec="none", tissue_pool="own", tissue_unroll=0,
                 reg_hidden=32, reg_depth=2, reg_kernel=1, attention_dense=False, tissue_soft=False):
        super().__init__()
        f = [init_filters * (2 ** i) for i in range(5)]
        cb = lambda a, b: ConvBlock(a, b, norm=norm)

        self.enc1 = cb(in_channels, f[0])
        self.enc2 = cb(f[0], f[1])
        self.enc3 = cb(f[1], f[2])
        self.enc4 = cb(f[2], f[3])
        self.bottleneck = cb(f[3], f[4])

        self.up4 = nn.ConvTranspose3d(f[4], f[3], kernel_size=2, stride=2)
        self.dec4 = cb(f[3] * 2, f[3])
        self.up3 = nn.ConvTranspose3d(f[3], f[2], kernel_size=2, stride=2)
        self.dec3 = cb(f[2] * 2, f[2])
        self.up2 = nn.ConvTranspose3d(f[2], f[1], kernel_size=2, stride=2)
        self.dec2 = cb(f[1] * 2, f[1])
        self.up1 = nn.ConvTranspose3d(f[1], f[0], kernel_size=2, stride=2)
        self.dec1 = cb(f[0] * 2, f[0])

        self.final_conv = nn.Conv3d(f[0], out_channels, kernel_size=1)
        self.regression_channels = regression_channels
        # seg_prior: the regression head also reads the segmentation head's
        # softmax, so it can judge intensity against the tissue it is in
        # rather than in isolation. Detached -- the prior is consumed, not
        # co-trained, so the regression loss cannot pull the anatomy head off
        # its own target. Soft rather than argmax so an uncertain head passes
        # a flat distribution, which is little information, not wrong
        # information.
        self.seg_prior = seg_prior
        # tissue_rel: one more head input channel from TissueRelative (no
        # parameters); the buffer marks it in the state_dict for detect_tissue_rel().
        self.tissue_rel = tissue_rel
        self.tissue_refs = tissue_refs if tissue_rel else 0
        self.tissue_vec = tissue_vec if tissue_rel else "none"
        self.tissue_pool = tissue_pool if tissue_rel else "own"
        self.tissue_soft = bool(tissue_soft) if tissue_rel else False
        if tissue_rel:
            self.tissue_relative = TissueRelative(refs=self.tissue_refs, vec=self.tissue_vec, pool=self.tissue_pool, soft=self.tissue_soft)
            self.register_buffer(TISSUE_REL_FLAG, torch.tensor(1 + self.tissue_refs + self.tissue_relative.n_vec))   # buffer value = channel count
            if self.tissue_soft:
                self.register_buffer(TISSUE_SOFT_FLAG, torch.tensor(1))   # marker only: channel count unchanged
            if self.tissue_pool != "own":
                self.register_buffer(TISSUE_POOL_FLAG, torch.tensor(TissueRelative.POOLS.index(self.tissue_pool)))   # marker only: channel count unchanged
        # unrolled re-estimation (2026-09-29): after a first pass, voxels whose |r| is a robust outlier over the brain
        # (|r| > median + 3 x 1.4826 MAD, per scan) are dropped from every pool and the channel + head are recomputed;
        # one shared head, loss on the final pass. The marker records the pass count for inference.
        self.tissue_unroll = int(tissue_unroll) if tissue_rel else 0
        if self.tissue_unroll:
            self.register_buffer(TISSUE_UNROLL_FLAG, torch.tensor(self.tissue_unroll))
        if reg_head not in ("linear", "mlp"):
            raise ValueError(f"reg_head must be 'linear' or 'mlp', got {reg_head!r}")
        self.reg_head = reg_head
        reg_in = f[0] + (out_channels if seg_prior else 0) + ((1 + self.tissue_refs + self.tissue_relative.n_vec) if tissue_rel else 0)
        # 'linear' keeps the regression_conv.* keys so old checkpoints load unchanged.
        self.regression_conv = self.regression_mlp = None
        self.reg_hidden, self.reg_depth, self.reg_kernel = int(reg_hidden), int(reg_depth), int(reg_kernel)
        if regression_channels > 0 and reg_head == "mlp":
            # default (32, 2, 1) = the original conv-ELU-conv head; reg_depth adds hidden 1x1 layers, reg_kernel widens the first layer's support
            layers = [nn.Conv3d(reg_in, self.reg_hidden, kernel_size=self.reg_kernel, padding=self.reg_kernel // 2), nn.ELU()]
            for _ in range(self.reg_depth - 2):
                layers += [nn.Conv3d(self.reg_hidden, self.reg_hidden, kernel_size=1), nn.ELU()]
            layers.append(nn.Conv3d(self.reg_hidden, regression_channels, kernel_size=1))
            self.regression_mlp = nn.Sequential(*layers)
        elif regression_channels > 0:
            self.regression_conv = nn.Conv3d(reg_in, regression_channels, kernel_size=1)
        self._trunk_frozen = False
        self.probability_channels = probability_channels
        self.probability_conv = (nn.Conv3d(f[0], probability_channels, kernel_size=1)
                                 if probability_channels > 0 else None)
        # Left as None by default so a model without attention keeps exactly
        # the parameter set earlier checkpoints were saved with.
        self.attention = attention
        self.attention_dense = bool(attention_dense)
        if attention:
            # dense: the gating signals for level i are u_i plus every coarser decoder feature. dec_{i+1} is NOT
            # listed because u_i = up_i(dec_{i+1}) already is it. So att4 is unchanged (its gate is the bottleneck),
            # att3 gains b, att2 gains d4 and b, att1 gains d3, d4 and b -- 1, 2, 3 and 4 gating inputs per level.
            xg = (lambda *c: tuple(c)) if self.attention_dense else (lambda *c: ())
            self.att4 = AttentionGate(f[3], f[3], identity_init=attention_identity_init, extra_gate_c=xg())
            self.att3 = AttentionGate(f[2], f[2], identity_init=attention_identity_init, extra_gate_c=xg(f[4]))
            self.att2 = AttentionGate(f[1], f[1], identity_init=attention_identity_init, extra_gate_c=xg(f[3], f[4]))
            self.att1 = AttentionGate(f[0], f[0], identity_init=attention_identity_init, extra_gate_c=xg(f[2], f[3], f[4]))

    def forward(self, x, return_features=False):
        """return_features=True also returns the bottleneck activation, for
        losses that compare encoder representations across two inputs.

        Return shape depends on which heads exist; the bottleneck, when
        requested, is always last:
          neither:       out                 (or (out, b))
          regression:    (out, reg)          (or (out, reg, b))
          both:          (out, reg, prob)    (or (out, reg, prob, b))
        prob is a sigmoid probability, not a logit.
        """
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, 2))
        e3 = self.enc3(F.max_pool3d(e2, 2))
        e4 = self.enc4(F.max_pool3d(e3, 2))
        b = self.bottleneck(F.max_pool3d(e4, 2))
        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, self.att4(u4, e4) if self.attention else e4], dim=1))
        u3 = self.up3(d4)
        _x3 = (b,) if self.attention_dense else ()
        d3 = self.dec3(torch.cat([u3, self.att3(u3, e3, _x3) if self.attention else e3], dim=1))
        u2 = self.up2(d3)
        _x2 = (d4, b) if self.attention_dense else ()
        d2 = self.dec2(torch.cat([u2, self.att2(u2, e2, _x2) if self.attention else e2], dim=1))
        u1 = self.up1(d2)
        _x1 = (d3, d4, b) if self.attention_dense else ()
        d1 = self.dec1(torch.cat([u1, self.att1(u1, e1, _x1) if self.attention else e1], dim=1))
        out = self.final_conv(d1)

        if self.regression_channels > 0:
            h = d1
            if self.seg_prior or self.tissue_rel:
                parts = [d1]
                if self.seg_prior:
                    parts.append(torch.softmax(out.detach(), dim=1))
                if self.tissue_rel:
                    parts.append(self.tissue_relative(x, out))
                h = torch.cat(parts, dim=1)
            reg = self.regression_head_module(h)
            for _ in range(self.tissue_unroll if self.tissue_rel else 0):
                excl = self._reg_outliers(reg.detach(), out.detach())
                h = torch.cat(parts[:-1] + [self.tissue_relative(x, out, exclude=excl)], dim=1)
                reg = self.regression_head_module(h)
            if self.probability_conv is not None:
                prob = torch.sigmoid(self.probability_conv(d1))
                return (out, reg, prob, b) if return_features else (out, reg, prob)
            return (out, reg, b) if return_features else (out, reg)
        return (out, b) if return_features else out

    @staticmethod
    def _reg_outliers(reg, logits):
        """Per scan: brain voxels whose |r| exceeds median + 3 x 1.4826 MAD of |r| over the brain (robust outlier rule, no fixed cut)."""
        a = reg[:, 0].abs().float(); cls = logits.argmax(dim=1)
        brain = ~torch.isin(cls, torch.tensor(TissueRelative.SKIP, device=cls.device))
        excl = torch.zeros_like(brain)
        for b in range(a.shape[0]):
            v = a[b][brain[b]]
            if v.numel() < 200:
                continue
            med = torch.quantile(v, 0.5); mad = torch.quantile((v - med).abs(), 0.5)
            excl[b] = brain[b] & (a[b] > med + 3.0 * 1.4826 * mad)
        return excl

    @property
    def regression_head_module(self):
        return self.regression_mlp if self.reg_head == "mlp" else self.regression_conv

    def regression_head_parameters(self):
        m = self.regression_head_module
        return list(m.parameters()) if m is not None else []

    def freeze_trunk(self):
        """requires_grad off on every parameter but the regression head's;
        trunk BatchNorm stays in eval mode under train() so its running
        statistics do not drift. Returns (n_trainable, n_total)."""
        head = {id(p) for p in self.regression_head_parameters()}
        for p in self.parameters():
            p.requires_grad_(id(p) in head)
        self._trunk_frozen = True
        self.train(self.training)
        return (sum(p.numel() for p in self.parameters() if p.requires_grad),
                sum(p.numel() for p in self.parameters()))

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "_trunk_frozen", False):
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
        return self


def load_checkpoint(ckpt_path, device, out_channels=56, verbose=True):
    """Build a model matching the checkpoint's norm layer and regression
    head (if any), and load it.

    Returns (model, state_dict, norm). The model is NOT set to eval() -- for
    BatchNorm the choice of mode changes the output substantially, so it is
    left to the caller.
    """
    ck = torch.load(ckpt_path, map_location=device)
    state = ck["model_state_dict"] if (isinstance(ck, dict) and "model_state_dict" in ck) else ck
    norm = detect_norm(state)
    in_channels = detect_in_channels(state)
    regression_channels = detect_regression_channels(state)
    probability_channels = detect_probability_channels(state)
    attention = detect_attention(state)
    seg_prior = detect_seg_prior(state)
    reg_head = detect_reg_head(state)
    tissue_rel = detect_tissue_rel(state)
    if verbose:
        name = "BatchNorm3d" if norm is batchnorm else norm.__name__
        print(f"[networks] checkpoint uses {name}"
              + (f", regression head ({regression_channels} ch)" if regression_channels else "")
              + (", mlp regression head" if reg_head == "mlp" else "")
              + (", tissue-relative channel" if tissue_rel else "")
              + (f", probability head ({probability_channels} ch)" if probability_channels else "")
              + (f", {in_channels} input channels" if in_channels != 1 else "")
              + (", skip attention" if attention else "")
              + (", seg prior into regression" if seg_prior else ""))
    model = model_from_state(state, device, out_channels)
    return model, state, norm


def model_from_state(state, device, out_channels=56):
    """Build the SynthSegUNet a state dict describes (every head option detected) and load it; shared by load_checkpoint and the
    downstream feature-input nets that carry a frozen trunk inside their own checkpoint (2026-10-09)."""
    norm = detect_norm(state)
    in_channels = detect_in_channels(state)
    regression_channels = detect_regression_channels(state)
    probability_channels = detect_probability_channels(state)
    attention = detect_attention(state)
    seg_prior = detect_seg_prior(state)
    reg_head = detect_reg_head(state)
    tissue_rel = detect_tissue_rel(state)
    if out_channels is None:                      # registration flow field
        out_channels = state["final_conv.weight"].shape[0]
    # 2026-10-06: the head options added since 09-26 (reference channels, class vector, pools, unroll, dense gates) were
    # detected for the log line but never passed to the constructor, so eval/fpr_healthy.py could not load a refs checkpoint
    # (regression_mlp.0.weight: 28 vs 25 input channels). Detect them all; compare_datasets builds its own model and was unaffected.
    model = SynthSegUNet(in_channels=in_channels, out_channels=out_channels, norm=norm,
                         regression_channels=regression_channels,
                         probability_channels=probability_channels,
                         attention=attention, attention_dense=detect_attention_dense(state), seg_prior=seg_prior,
                         reg_head=reg_head, tissue_rel=tissue_rel, tissue_refs=detect_tissue_refs(state),
                         tissue_vec=detect_tissue_vec(state), tissue_pool=detect_tissue_pool(state),
                         tissue_unroll=detect_tissue_unroll(state), tissue_soft=detect_tissue_soft(state)).to(device)
    model.load_state_dict(state)
    return model


def load_checkpoint_into(model, state, strict_trunk=True, log=print):
    """Load a state_dict into an already-built model, re-initialising the
    regression head when the checkpoint's head does not match the model's
    (an old linear checkpoint into an mlp / tissue_rel model). The rest
    loads strictly when strict_trunk. Returns True if the head was left at
    its fresh initialisation."""
    shapes = lambda sd: {k: tuple(v.shape) for k, v in sd.items() if is_regression_head_key(k)}
    if shapes(state) == shapes(model.state_dict()):
        model.load_state_dict(state, strict=strict_trunk)
        return False
    res = model.load_state_dict({k: v for k, v in state.items() if not is_regression_head_key(k)},
                                strict=False)
    missing = [k for k in res.missing_keys if not is_regression_head_key(k)]
    if strict_trunk and (missing or res.unexpected_keys):
        raise RuntimeError(f"trunk mismatch: missing {missing[:5]}, unexpected {list(res.unexpected_keys)[:5]}")
    log("regression head re-initialised (checkpoint head: "
        f"{head_description(detect_reg_head(state), detect_tissue_rel(state), detect_tissue_refs(state), detect_tissue_vec(state), detect_tissue_pool(state))}, "
        f"requested: {head_description(model.reg_head, model.tissue_rel, model.tissue_refs, getattr(model, 'tissue_vec', 'none'), getattr(model, 'tissue_pool', 'own'))})")
    return True