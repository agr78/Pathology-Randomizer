"""Regression-target helpers shared by train.py and eval/test_train_flags.py."""
import torch


def apply_target_region(target, lesion_mask, dI, floor, eps=1e-6):
    """--target_region: inside lesion_mask, target = sign * max(|target|, floor).

    sign = sign(dI) per voxel; |dI| < eps takes the sign of that sample's mean
    dI over the mask (0 if the mean is exactly 0). Outside the mask nothing
    changes. Per sample along dim 0. floor <= 0 returns target unchanged.
    """
    if floor <= 0:
        return target
    m = lesion_mask.reshape(target.shape).to(torch.bool)
    dI = dI.reshape(target.shape).to(target.dtype)
    n = target.shape[0]
    mf = m.reshape(n, -1).to(target.dtype)
    mean_dI = (dI.reshape(n, -1) * mf).sum(1) / mf.sum(1).clamp(min=1.0)
    mean_sign = torch.sign(mean_dI).reshape((n,) + (1,) * (target.dim() - 1)).expand_as(target)
    sign = torch.where(dI.abs() < eps, mean_sign, torch.sign(dI))
    return torch.where(m, sign * target.abs().clamp(min=floor), target)


# --- regression-head-only training (--freeze_trunk / --reg_head / --tissue_rel) ---
from torch.nn.modules.batchnorm import _BatchNorm  # noqa: E402

REG_HEAD_PREFIXES = ("regression_conv.", "regression_mlp.")
REG_HEAD_MARKERS = ("tissue_rel_flag",)


def is_reg_head_key(name):
    """Parameter / state-dict name belonging to the regression head or its tissue_rel marker."""
    return name.startswith(REG_HEAD_PREFIXES) or name in REG_HEAD_MARKERS


def describe_reg_head(state_dict):
    """'linear' | 'mlp' | 'none', plus '+tissue_rel' when the marker buffer is present."""
    keys = list(state_dict)
    if any(k.startswith("regression_mlp.") for k in keys):
        head = "mlp"
    elif any(k.startswith("regression_conv.") for k in keys):
        head = "linear"
    else:
        return "none"
    if not any(k in REG_HEAD_MARKERS for k in keys):
        return head
    n = int(state_dict["tissue_rel_flag"]) if "tissue_rel_flag" in state_dict else 1   # channel count: 1 own-class + refs + class-vector channels
    from networks import _TR_LAYOUT
    refs, vec = _TR_LAYOUT.get(n, (0, "none"))
    from networks import detect_tissue_pool, detect_tissue_unroll, detect_reg_arch
    pool = detect_tissue_pool(state_dict); unroll = detect_tissue_unroll(state_dict); hid, dep, ker = detect_reg_arch(state_dict)
    arch = f"({hid}x{dep}, k{ker})" if head == "mlp" and (hid, dep, ker) != (32, 2, 1) else ""
    return (head + arch + "+tissue_rel" + (f"+pool:{pool}" if pool != "own" else "") + (f"+unroll{unroll}" if unroll else "")
            + (f"+refs{refs}" if refs else "") + (f"+vec{vec}" if vec != "none" else ""))


def reg_head_matches(ckpt_state, model_state):
    """True when the checkpoint's head tensors (names and shapes) equal the model's."""
    def head(sd):
        return {k: tuple(v.shape) for k, v in sd.items() if is_reg_head_key(k)}
    return head(ckpt_state) == head(model_state)


def expand_final_conv(state_dict, model_state):
    """A checkpoint with FEWER anatomy classes than the model (--choroid_target on a 56-class checkpoint): copy its
    final_conv rows and keep the model's fresh initialisation for the appended classes. Returns (state_dict, expanded)."""
    k_w, k_b = "final_conv.weight", "final_conv.bias"
    if k_w in state_dict and k_w in model_state and state_dict[k_w].shape[0] < model_state[k_w].shape[0] \
            and state_dict[k_w].shape[1:] == model_state[k_w].shape[1:]:
        n = state_dict[k_w].shape[0]; sd = dict(state_dict)
        w = model_state[k_w].clone(); w[:n] = state_dict[k_w]; sd[k_w] = w
        if k_b in sd:
            b = model_state[k_b].clone(); b[:n] = state_dict[k_b]; sd[k_b] = b
        return sd, True
    return state_dict, False


def load_weights_keep_head(model, state_dict):
    """Strict load when the checkpoint's regression head matches the model's; otherwise load
    everything but the head strictly and keep the model's fresh head.
    Returns (reinit, checkpoint head, model head) as bools/strings."""
    model_state = model.state_dict()
    state_dict, expanded = expand_final_conv(state_dict, model_state)   # appended anatomy classes: fresh rows, Adam moments restart
    ckpt_head, model_head = describe_reg_head(state_dict), describe_reg_head(model_state)
    fresh_gates = getattr(model, "attention", False) and not any(k.startswith("att") for k in state_dict)
    # --tissue_pool on a checkpoint without the marker: same head, same channel count, only the pools change -> warm head,
    # the marker buffer is the one allowed missing key (like fresh attention gates)
    NEW_MARKERS = ("tissue_pool_flag", "tissue_unroll_flag", "tissue_soft_flag")   # tissue_soft_flag added 2026-10-07 (soft own-class channel: same head, same channel count)   # same head, same channel count -> warm head; the marker buffers may be new
    new_marker = any(k in model_state and k not in state_dict for k in NEW_MARKERS)
    if reg_head_matches(state_dict, model_state):
        if not fresh_gates and not new_marker:
            model.load_state_dict(state_dict)
            return expanded, ckpt_head, model_head
        # --attention on a gate-less checkpoint: trunk AND head load strictly, only the att*.{wg,wx,psi} keys are new
        res = model.load_state_dict(state_dict, strict=False)
        stray = [k for k in res.missing_keys if not (fresh_gates and k.startswith("att")) and not (new_marker and k in NEW_MARKERS)]
        if res.unexpected_keys or stray:
            raise RuntimeError(f"checkpoint/model mismatch beyond the attention gates: unexpected {list(res.unexpected_keys)[:5]} missing {stray[:5]}")
        return expanded, ckpt_head, model_head
    rest = {k: v for k, v in state_dict.items() if not is_reg_head_key(k)}
    res = model.load_state_dict(rest, strict=False)
    # a gate-less checkpoint forked with --attention: the att*.{wg,wx,psi} keys are new parameters (fresh gates),
    # not a mismatch -- everything else must still load strictly
    stray = [k for k in res.missing_keys if not is_reg_head_key(k) and not (getattr(model, "attention", False) and k.startswith("att")) and k not in ("tissue_pool_flag", "tissue_unroll_flag")]
    if res.unexpected_keys or stray:
        raise RuntimeError(f"checkpoint/model mismatch outside the regression head: "
                           f"unexpected {list(res.unexpected_keys)[:5]} missing {stray[:5]}")
    return True, ckpt_head, model_head


def freeze_trunk(model):
    """requires_grad=False on every parameter outside the regression head.
    Returns (trainable parameters, n_trainable, n_total) with counts in elements."""
    trainable, n_tr, n_tot = [], 0, 0
    for name, p in model.named_parameters():
        keep = is_reg_head_key(name)
        p.requires_grad_(keep)
        n_tot += p.numel()
        if keep:
            trainable.append(p)
            n_tr += p.numel()
    if not trainable:
        raise ValueError("freeze_trunk: no regression-head parameters to train")
    return trainable, n_tr, n_tot


def frozen_bn_batch(model):
    """Frozen BatchNorm layers in train mode with momentum 0: the forward uses the batch's own
    statistics (the batch-statistics readout) and the running statistics never move. Returns the count."""
    n = 0
    for m in model.modules():
        if isinstance(m, _BatchNorm) and not any(p.requires_grad for p in m.parameters(recurse=False)):
            m.train()
            m.momentum = 0.0
            n += 1
    return n


def frozen_bn_eval(model):
    """eval() on every BatchNorm with no trainable parameter, so its running statistics stay
    fixed while model.training stays True. Call after each model.train(). Returns the count."""
    n = 0
    for m in model.modules():
        if isinstance(m, _BatchNorm) and not any(p.requires_grad for p in m.parameters(recurse=False)):
            m.eval()
            n += 1
    return n
