"""Registration on a REAL held-out pool cohort (2026-10-09): e.g. ABIDE T1w from the curated pool, which never enters training
(the allowlist is Chinese-HCP + HCP, plus opt-in elderly cohorts). Two things the lesion datasets cannot give:
  1. base registration quality on real anatomy with REAL labels: the curated segmentation (<case>.brainseg_with_extracerebral.nii,
     project S scheme, mapped to compact indices) of both subjects -- no healthy-SynthSeg proxy, no lesion.
  2. --paint: a synthetic composition lesion painted INTO the real moving scan (random WM blob, intensity = the scan's own CSF or GM
     median with a soft edge), so the lesion mask and the labels under it are exact on real anatomy. Reported like
     eval/register_shell_val.py: brain / shell / lesion Dice and the lesion SENSITIVITY |flow(painted) - flow(clean)| in voxels.
Pairs are seeded; models = identity + the given checkpoints (healthy-prior / pathology-prior / consistency arms).
    cd <snapshot> && python eval/register_eval_pool.py --cohort ABIDE --ckpts A.pth B.pth C.pth --n 40 --paint \
        --out <csv>
2026-10-14 (the 10-13 ABIDE pass: 15/40 pairs whose single-start affine never converged, identity dice_large 0.19-0.37 vs 0.55-0.76;
both heads over-deformed at vel_scale 0.1). The default run is numerically identical to 10-13 for every existing column except
through --bn_mode (default batch; --bn_mode running reproduces 10-13); the new columns are appended after them.
  --affine_robust 1 (default): robust_affine_to -- the head-moments start of 10-13, then the BRAIN-moments start (centroid and
      spread of the large structures only: the extracerebral block biases the head centroid by tens of voxels when two scans cover
      the neck differently, the failure mode seen on the stand-in pool), then the head start with iters=600/lr=1e-2, then the six
      +-10 degree single-axis rotations of the brain start; stops at the first whose soft label Dice reaches --affine_min (0.45),
      else keeps the best start and writes the pair with affine_ok=0 (never skipped). --affine_robust 0 = the old single start.
  --vel_scales s1 s2 ... (default 1.0): the velocity is computed ONCE per model and integrated per factor s (v * vel_scale * s);
      one row per (pair, model, scale), column 'scale'; the identity is written once, at scale 1.0.
  --pool DIR: the pool directory (default = the curated pool), e.g. a synthetic stand-in where ABIDE is not mounted.
  columns: scale, affine_dice, affine_ok, n_starts, identity_dice_before_affine (dice_large after centring only, the regime of the
      pair), u_affine_share (fraction of sum|u|^2 over the fixed brain explained by a least-squares 12-dof affine of u, see
      u_affine_stats), u_translation (|mean u| over the fixed brain, voxels) -- the head's resting global field, per pair.
  --bn_mode batch|running (default batch): BatchNorm statistics of the loaded nets (td.load_task_model). batch = each volume
      normalised by its own statistics (compare_datasets.bn_batch_mode), the convention of the detection readout and of the
      fine-tune validation; running = the running averages (model.eval()), which reproduces the rows written before this flag.
      Trailing CSV column bn_mode on every row (identity rows carry the run's mode).
"""
import argparse, csv, glob, os, sys, time
import numpy as np, nibabel as nib, torch, torch.nn.functional as F
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_downstream as td   # noqa: E402
from networks import SynthSegUNet, detect_attention, detect_attention_dense  # noqa: E402
from compare_datasets import LABEL_LIST_SEGMENTATION  # noqa: E402

POOL = os.environ.get('PR_DATA_DIR', '/path/to/data_dir')
FORBIDDEN = ("atlas", "brats", "wmh", "isles")
LUT = np.zeros(max(LABEL_LIST_SEGMENTATION) + 1, np.int64)
for i, l in enumerate(LABEL_LIST_SEGMENTATION): LUT[l] = i      # raw S id -> compact channel; ids not in the list -> 0
WM_RAW, GM_RAW, CSF_RAW = (1, 18), (2, 19), (16,)
# large-structure set for the second Dice column (literature convention): cerebral WM + cortex, lateral ventricles, cerebellar WM +
# cortex, thalamus, caudate, putamen, pallidum, hippocampus, amygdala, brainstem -- compact indices
LARGE = sorted(LUT[i] for i in (1, 18, 2, 19, 3, 20, 5, 22, 6, 23, 7, 24, 8, 25, 9, 26, 10, 27, 14, 28, 15, 29, 13))


def load(t1, dev):
    im = nib.load(t1); x = np.asanyarray(im.dataobj).astype(np.float32).squeeze()
    sp = t1[:-len("T1w.nii")] + "brainseg_with_extracerebral.nii"
    if not os.path.isfile(sp): sp = t1[:-len("T1w.nii")] + "brainseg.nii"
    if not os.path.isfile(sp): return None
    sg = np.asanyarray(nib.load(sp).dataobj).squeeze().astype(np.int64)
    if sg.shape != x.shape: return None
    if not (0.10 <= np.isin(sg, WM_RAW).sum() / max((sg > 0).sum(), 1) <= 0.60): return None   # wrong ids guard
    lab = LUT[np.clip(sg, 0, LUT.size - 1)]
    return dict(img=torch.from_numpy(x).to(dev), lab=torch.from_numpy(lab).to(dev), raw=sg, zooms=im.header.get_zooms()[:3])


def paint(c, rng, dev):
    """Composition lesion on real anatomy: a smoothed random blob inside WM, intensity = the scan's own CSF or GM median (coin),
    soft edge; returns (painted image, lesion mask). Target volume 2-30 mL drawn log-uniform."""
    x = c["img"].cpu().numpy(); raw = c["raw"]; wm = np.isin(raw, WM_RAW)
    vol_vox = float(np.exp(rng.uniform(np.log(2000), np.log(30000))))
    r = (3 * vol_vox / (4 * np.pi)) ** (1 / 3)
    idx = np.argwhere(ndimage.binary_erosion(wm, iterations=3)); cen = idx[rng.integers(len(idx))]
    g = np.indices(x.shape).astype(np.float32); d = np.sqrt(((g - cen[:, None, None, None]) ** 2 * rng.uniform(0.6, 1.6, 3)[:, None, None, None]).sum(0))
    noise = ndimage.gaussian_filter(rng.standard_normal(x.shape).astype(np.float32), 6); noise /= noise.std() + 1e-6
    blob = (d + 0.35 * r * noise) < r
    blob &= ndimage.binary_dilation(wm, iterations=2)                 # stay in and just around WM
    soft = np.clip(ndimage.gaussian_filter(blob.astype(np.float32), 1.0), 0, 1)
    tgt = np.median(x[np.isin(raw, CSF_RAW if rng.random() < 0.5 else GM_RAW)])
    y = x * (1 - soft) + tgt * soft
    return torch.from_numpy(y.astype(np.float32)).to(dev), torch.from_numpy(blob).to(dev)


def centre(c, keys):
    m = c["lab"] > 0; idx = torch.nonzero(m).float().mean(0); ctr = (torch.tensor(m.shape, device=idx.device) - 1) / 2
    sh = [int(round(float(ctr[i] - idx[i]))) for i in range(3)]
    return {k: (torch.roll(v, shifts=sh, dims=(0, 1, 2)) if k in keys else v) for k, v in c.items()}


def fit(t, shape):
    """Centre-pad (zeros) a (D,H,W) tensor onto a common shape: pool scans differ in matrix size between subjects."""
    pads = []
    for s, S in zip(reversed(t.shape), reversed(shape)): r = S - s; pads += [r // 2, r - r // 2]
    return F.pad(t, pads)


def _moments(m):
    idx = torch.nonzero(m).float(); return idx.mean(0), idx.std(0)


def moments_init(A, B):
    """The moments start of the affine, (M0, t0) in affine_to's parametrisation: per-axis scale of the label extents, zero
    translation (the centroids are matched inside the parametrisation itself)."""
    (_, sm), (_, sf) = _moments(A["lab"] > 0), _moments(B["lab"] > 0)
    return torch.diag(sm / sf), torch.zeros(3, device=sm.device)


def brain_moments_init(A, B, R=None):
    """Moments start from the LARGE (brain) structures only (2026-10-14), optionally rotated by R about the brain centroid, in
    affine_to's head-centroid parametrisation (t0 carries the brain centroids onto each other). The head moments are biased by the
    extracerebral block when the scans cover the neck differently: stand-in pair 1 had its brain 35 vox off the head centroid, every
    head-moments start stuck at soft Dice 0.28, this start reached 0.53 (and 0.56 vs 0.52 on the pair the head start solved)."""
    large = torch.tensor(LARGE, device=A["lab"].device)
    (cm, _), (cf, _) = _moments(A["lab"] > 0), _moments(B["lab"] > 0)
    (bm, sm), (bf, sf) = _moments(torch.isin(A["lab"], large)), _moments(torch.isin(B["lab"], large))
    M0 = torch.diag(sm / sf) if R is None else R @ torch.diag(sm / sf)
    return M0, (bm - cm) + M0 @ (cf - bf)


def rot_axis(axis, deg, dev):
    """Rotation by `deg` degrees about index axis `axis` (0, 1, 2 = D, H, W) of a (d, h, w) offset -- a start for affine_to (2026-10-14)."""
    c, s = float(np.cos(np.radians(deg))), float(np.sin(np.radians(deg))); i, j = [k for k in range(3) if k != axis]
    R = torch.eye(3, device=dev); R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def affine_to(A, B, keys, iters=300, lr=2e-2, down=4, init=None):
    """Full 12-parameter affine of the moving scan onto the fixed one, fitted by gradient descent on the LABEL MAPS: maximise the
    mean soft Dice of the large structures between the affinely resampled moving one-hot and the fixed one-hot, on a `down`x
    coarser grid (seconds on a GPU). Initialised from the moments solution (centroid translation + per-axis scale). Learning-free,
    deterministic, and the same for every arm; the label maps are the curated ones on ABIDE (exact), the proxies elsewhere.
    init=(M0, t0) (2026-10-14): start from these instead of the moments solution, in the same parametrisation
    (moving = cm + M (fixed - cf) + t, voxels); None keeps the default exactly. out['affine_dice'] = the soft Dice at the end."""
    dev = A["lab"].device; D, H, W = B["lab"].shape; Dm, Hm, Wm = A["lab"].shape
    cm, sm = _moments(A["lab"] > 0); cf, sf = _moments(B["lab"] > 0)
    M0, t0 = (torch.diag(sm / sf), torch.zeros(3, device=dev)) if init is None else (init[0].detach().float().to(dev), init[1].detach().float().to(dev))
    M = M0.clone().requires_grad_(True); t = t0.clone().requires_grad_(True)
    labels = [l for l in LARGE if (A["lab"] == l).any() and (B["lab"] == l).any()]
    def onehot(lab, labs): return torch.stack([(lab == l).float() for l in labs])[None]
    om = F.avg_pool3d(onehot(A["lab"], labels), down); of = F.avg_pool3d(onehot(B["lab"], labels), down)
    gd = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev) for n in of.shape[2:]], indexing="ij"), -1).float() * down + (down - 1) / 2   # fixed voxel coords
    def grid(Mx, tx, g, mshape):
        pm = cm + (g - cf) @ Mx.T + tx                                                                     # fixed voxel -> moving voxel
        return torch.stack([2 * (pm[..., 2] + 0.5) / mshape[2] - 1, 2 * (pm[..., 1] + 0.5) / mshape[1] - 1, 2 * (pm[..., 0] + 0.5) / mshape[0] - 1], -1)[None]
    opt = torch.optim.Adam([M, t], lr=lr)
    for it in range(iters):
        w = F.grid_sample(om, grid(M, t, gd, (Dm, Hm, Wm)), mode="bilinear", padding_mode="zeros", align_corners=False)   # normalised coords are resolution-free
        dice = (2 * (w * of).sum((2, 3, 4)) + 1) / (w.sum((2, 3, 4)) + of.sum((2, 3, 4)) + 1)
        loss = 1 - dice.mean(); opt.zero_grad(); loss.backward(); opt.step()
    M, t = M.detach(), t.detach()
    gf = torch.stack(torch.meshgrid(torch.arange(D, device=dev), torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij"), -1).float()
    norm = grid(M, t, gf, (Dm, Hm, Wm))
    out = dict(A)
    for k in keys:
        if A.get(k) is None: continue
        mode = "nearest" if k in ("lab", "les") else "bilinear"
        v = A[k].float()[None, None]
        out[k] = F.grid_sample(v, norm, mode=mode, padding_mode="zeros", align_corners=False)[0, 0]
        if k == "lab": out[k] = out[k].long()
        if k == "les": out[k] = out[k] > 0.5
    out["affine_dice"] = float(1 - loss.detach())
    return out


def robust_affine_to(A, B, keys, min_dice=0.45):
    """Multi-start affine_to (2026-10-14: 15/40 ABIDE pairs of 10-13 had a single moments start that never converged). Starts, in
    order: 1 the head-moments init as affine_to (so a pair it solves keeps the 10-13 numbers); 2 brain_moments_init; 3 the head
    init with iters=600, lr=1e-2; 4-9 the six +-10 degree single-axis rotations of the brain init. Stops at the first start whose
    affine_dice reaches min_dice, else returns the best of all nine.
    Returns (out, n_starts); the caller flags out['affine_dice'] < min_dice (affine_ok=0), it never skips the pair."""
    dev = A["lab"].device
    starts = [dict(), dict(init=brain_moments_init(A, B)), dict(iters=600, lr=1e-2)] + [dict(init=brain_moments_init(A, B, rot_axis(ax, sg * 10.0, dev))) for ax in range(3) for sg in (1, -1)]
    best = None
    for n, kw in enumerate(starts, 1):
        out = affine_to(A, B, keys, **kw)
        if best is None or out["affine_dice"] > best["affine_dice"]: best = out
        if out["affine_dice"] >= min_dice: break
    return best, n


def pad16(t):
    pads = []
    for s in reversed(t.shape[-3:]): r = (-s) % 16; pads += [r // 2, r - r // 2]
    return F.pad(t, pads)


def dice_region(wl, fl, region, labels):
    ds = []
    for l in labels:
        a = (wl == l) & region; b = (fl == l) & region; den = int(a.sum() + b.sum())
        if den: ds.append(2 * int((a & b).sum()) / den)
    return float(np.mean(ds)) if ds else float("nan")


@torch.no_grad()
def velocity_of(model, mov, fix):
    """The head's velocity on the pad16 grid, computed ONCE per model (2026-10-14; flow_from integrates it per scale); (None, shape)
    for the identity."""
    x = pad16(torch.stack([mov, fix])[None])
    if model is None: return None, x.shape[2:]
    with torch.amp.autocast("cuda", enabled=x.is_cuda): v = model(x)
    return v, x.shape[2:]


def flow_from(v, scale, a, pshape, dev):
    """Integrate v * vel_scale * scale (scale = the evaluation-time factor of --vel_scales); None -> the zero field."""
    if v is None: return torch.zeros((1, 3) + tuple(pshape), device=dev)
    return td.integrate_svf(v.float() * a.vel_scale * scale, a.int_steps)


@torch.no_grad()
def flow_of(model, mov, fix, a, scale=1.0):
    v, psh = velocity_of(model, mov, fix)
    return flow_from(v, scale, a, psh, mov.device), psh


def vox(flow):
    D, H, W = flow.shape[2:]
    return (flow * torch.tensor([W / 2, H / 2, D / 2], device=flow.device, dtype=flow.dtype).view(1, 3, 1, 1, 1)).norm(dim=1)[0]


def u_affine_stats(flow, brain):
    """The resting global field of a head (2026-10-14). Least-squares 12-dof affine u ~ A x + b of the voxel displacement over the
    fixed-brain voxels (x = voxel coordinates): returns (fraction of the field energy sum|u|^2 explained by the fit -- 1 = a pure
    global affine, 0 = nothing global; nan for the zero field --, norm of the mean displacement, voxels).
    flow (1,3,D,H,W) in normalised units with (x,y,z) channels as td.warp, brain (D,H,W) bool on the same grid."""
    D, H, W = flow.shape[2:]
    u = (flow[0] * torch.tensor([W / 2, H / 2, D / 2], device=flow.device, dtype=flow.dtype).view(3, 1, 1, 1))[:, brain].T.double()   # (N,3) voxels
    x = torch.nonzero(brain)[:, [2, 1, 0]].double(); x = torch.cat([x - x.mean(0), torch.ones(len(x), 1, dtype=x.dtype, device=x.device)], 1)   # (N,4)
    sol = torch.linalg.solve(x.T @ x, x.T @ u)                      # normal equations, centred coords: well conditioned
    e = float((u ** 2).sum())
    return (float(((x @ sol) ** 2).sum()) / e if e > 0 else float("nan")), float(u.mean(0).norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="ABIDE", help="curated-filename prefix"); ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--paint", action="store_true")
    ap.add_argument("--shell", type=int, default=10); ap.add_argument("--vel_scale", type=float, default=0.1); ap.add_argument("--int_steps", type=int, default=7)
    ap.add_argument("--out", default=None); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--affine", type=int, default=1, help="1: moments affine pre-alignment of the moving scan onto the fixed (translation + per-axis scale); the identity row then reports the affine-only result")
    ap.add_argument("--pool", default=POOL, help="pool directory holding <cohort>*T1w.nii and the curated brainseg companions (2026-10-14; default = the curated pool)")
    ap.add_argument("--affine_robust", type=int, default=1, help="1: robust_affine_to, multi-start affine (2026-10-14); 0: the single moments start of 10-13")
    ap.add_argument("--affine_min", type=float, default=0.45, help="soft label Dice a start must reach to stop retrying; a pair still below it after every start is written with affine_ok=0")
    ap.add_argument("--vel_scales", type=float, nargs="+", default=[1.0], help="evaluation-time factors applied to v (on top of --vel_scale) before integration: one row per factor, column 'scale'; identity at 1.0 only")
    ap.add_argument("--bn_mode", default="batch", choices=("batch", "running"), help="BatchNorm statistics at inference (2026-10-14): batch = every BN layer normalises the volume being inferred by its own statistics (compare_datasets.bn_batch_mode, momentum 0; the convention of the detection readout and of the fine-tune validation; default), running = the checkpoint's running averages (model.eval(): the rows written before 2026-10-14). Recorded in every CSV row (trailing column bn_mode). Feature-input nets (FeatureHeadNet) are unaffected: trunk on batch statistics, head on running statistics in both modes.")
    a = ap.parse_args(); dev = torch.device(a.device)
    if any(b in a.cohort.lower() for b in FORBIDDEN): raise SystemExit(f"REFUSED: {a.cohort} is evaluation data")
    files = sorted(glob.glob(os.path.join(a.pool, f"{a.cohort}*T1w.nii")))
    if len(files) < 2: raise SystemExit(f"no {a.cohort}*T1w.nii under {a.pool}")
    rng = np.random.default_rng(a.seed); order = rng.permutation(len(files))
    pairs = [(files[order[2 * i]], files[order[2 * i + 1]]) for i in range(min(a.n, len(files) // 2))]
    models = {"identity": None}
    for p in a.ckpts:
        models[p.rstrip("/").split("/")[-4] + "/" + os.path.basename(p)] = td.load_task_model(p, dev, bn_mode=a.bn_mode)   # BN convention 2026-10-14
    print(f"[register_eval_pool] {a.cohort}: {len(files)} scans, {len(pairs)} pairs, paint={a.paint}, models {list(models)}, scales {a.vel_scales}, affine_robust={a.affine_robust}, bn_mode {a.bn_mode}", flush=True)
    rows = []; fh = open(a.out, "w", newline="") if a.out else None; wr = None
    for i, (tm, tf) in enumerate(pairs):
        A, B = load(tm, dev), load(tf, dev)
        if A is None or B is None: print(f"skip pair {i} (no usable segmentation)", flush=True); continue
        les = None
        if a.paint:
            A["img_p"], A["les"] = paint(A, np.random.default_rng([a.seed, i]), dev)
        A = centre(A, ("img", "lab", "img_p", "les") if a.paint else ("img", "lab")); B = centre(B, ("img", "lab"))
        les = A.pop("les", None)   # 2026-10-14 review: rolled WITH the image; centring les against the already-centred lab left it in the old frame (13 vox off the painted blob on an off-centre stand-in)
        shape = tuple(max(A["lab"].shape[k], B["lab"].shape[k]) for k in range(3))   # common grid for the pair
        for k in ("img", "lab", "img_p"):
            if k in A: A[k] = fit(A[k], shape)
            if k in B: B[k] = fit(B[k], shape)
        if les is not None: les = fit(les, shape)
        id0 = dice_region(A["lab"], B["lab"], B["lab"] > 0, [l for l in LARGE if (A["lab"] == l).any() or (B["lab"] == l).any()])   # centred only (2026-10-14)
        aff = dict(affine_dice=float("nan"), affine_ok=1, n_starts=0)
        if a.affine:
            A["les"] = les
            if a.affine_robust: A, ns = robust_affine_to(A, B, ("img", "lab", "img_p", "les"), a.affine_min)
            else: A, ns = affine_to(A, B, ("img", "lab", "img_p", "les")), 1
            les = A.pop("les"); aff = dict(affine_dice=A["affine_dice"], affine_ok=int(A["affine_dice"] >= a.affine_min), n_starts=ns)
            if not aff["affine_ok"]: print(f"pair {i}: affine did not converge (soft dice {A['affine_dice']:.3f} after {ns} starts), written with affine_ok=0", flush=True)
            if les is not None and not les.any(): print(f"pair {i}: painted lesion left the grid after the affine, skipped", flush=True); continue
        ml, fl = A["lab"], B["lab"]; brain = fl > 0
        large = [l for l in LARGE if (ml == l).any() or (fl == l).any()]
        labels = [l for l in torch.unique(torch.cat([ml.flatten(), fl.flatten()])).tolist() if l != 0 and l not in td.EXTRACEREBRAL]
        fix = td.normalize(B["img"])[0]; mov_c = td.normalize(A["img"])[0]
        shell = F.max_pool3d(les.float()[None, None], 2 * a.shell + 1, stride=1, padding=a.shell)[0, 0] > 0 if les is not None else None
        brain_p = pad16(brain.float()) > 0.5                                       # the fixed brain on the flow grid (u_affine_stats)
        for name, m in models.items():
            t0 = time.time(); vc, psh = velocity_of(m, mov_c, fix)
            vp = velocity_of(m, td.normalize(A["img_p"])[0], fix)[0] if les is not None else None
            unpad = lambda t: t[tuple(slice((p - s) // 2, (p - s) // 2 + s) for p, s in zip(psh, ml.shape))]
            for s in ([1.0] if m is None else a.vel_scales):                       # one row per scale; the identity once
                fc = flow_from(vc, s, a, psh, dev)
                r = dict(pair=i, moving=os.path.basename(tm)[:30], fixed=os.path.basename(tf)[:30], model=name, sec=0.0)
                if les is None:
                    wl = unpad(td.warp(pad16(ml.float()[None, None]), fc, mode="nearest")[0, 0]).long(); u = unpad(vox(fc))
                    r.update(dice_brain=dice_region(wl, fl, brain, labels), dice_large=dice_region(wl, fl, brain, large), u_brain=float(u[brain].mean()), fold=float(td.jacobian_fold_fraction(fc)))
                    share, trans = u_affine_stats(fc, brain_p)
                else:
                    fp = flow_from(vp, s, a, psh, dev)
                    wl = unpad(td.warp(pad16(ml.float()[None, None]), fp, mode="nearest")[0, 0]).long()
                    wsh = unpad(td.warp(pad16(shell.float()[None, None]), fp, mode="nearest")[0, 0]) > 0; wles = unpad(td.warp(pad16(les.float()[None, None]), fp, mode="nearest")[0, 0]) > 0
                    up, dv = unpad(vox(fp)), unpad(vox(fp - fc))
                    r.update(dice_brain=dice_region(wl, fl, brain, labels), dice_large=dice_region(wl, fl, brain, large), dice_shell=dice_region(wl, fl, wsh & brain, labels), dice_lesion=dice_region(wl, fl, wles & brain, labels),
                             u_lesion=float(up[les].mean()), u_brain=float(up[brain].mean()), sens_lesion=float(dv[les].mean()), sens_shell=float(dv[shell & brain].mean()),
                             fold=float(td.jacobian_fold_fraction(fp)), lesion_mL=float(les.sum()) * float(np.prod(A["zooms"])) / 1000.0)
                    share, trans = u_affine_stats(fp, brain_p)
                r.update(scale=s, **aff, identity_dice_before_affine=id0, u_affine_share=share, u_translation=trans)   # appended columns (2026-10-14)
                r["bn_mode"] = a.bn_mode   # trailing column (2026-10-14)
                r["sec"] = round(time.time() - t0, 1); t0 = time.time(); rows.append(r)
                if fh and wr is None: wr = csv.DictWriter(fh, fieldnames=list(r)); wr.writeheader()
                if wr: wr.writerow(r); fh.flush()
                print(f"pair {i:3d} {name[:34]:34s} " + " ".join(f"{k} {v:.3f}" for k, v in r.items() if isinstance(v, float) and k != "sec"), flush=True)
    cols = [c for c in ("dice_brain", "dice_large", "dice_shell", "dice_lesion", "u_lesion", "u_brain", "sens_lesion", "sens_shell", "fold", "u_affine_share", "u_translation") if rows and c in rows[0]]
    med = lambda xs: float(np.nanmedian(xs)) if np.isfinite(xs).any() else float("nan")
    n_ok = sum(r["affine_ok"] for r in rows if r["model"] == "identity")
    print(f"\n# {a.cohort}, medians over {len({r['pair'] for r in rows})} pairs" + (" (painted lesions)" if a.paint else "") + (f" -- 'identity' = affine pre-alignment only, affine_ok {n_ok}/{len({r['pair'] for r in rows})}" if a.affine else ""))
    wd = {c: max(12, len(c) + 1) for c in cols}
    print(f"{'model':36s}" + "".join(f"{c:>{wd[c]}s}" for c in cols))
    for name, s in dict.fromkeys((r["model"], r["scale"]) for r in rows):
        rr = [r for r in rows if r["model"] == name and r["scale"] == s]
        print(f"{(name[:30] + f' @{s:.2f}' if len(a.vel_scales) > 1 else name[:36]):36s}" + "".join(f"{med(np.array([r[c] for r in rr], float)):{wd[c]}.3f}" for c in cols))


if __name__ == "__main__":
    main()
