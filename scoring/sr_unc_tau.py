#!/usr/bin/env python
"""2026-11-11: gate threshold for --lesion_gate unc from the arm's OWN clean validation draws -- no real data, no ground truth.
Loads an --sr_uncertainty checkpoint, draws N clean (and pathological) synthetic cases exactly as the trainer's validation does, runs the
net once per draw and reports quantiles of b = exp(log b) over the brain of the CLEAN rendering: tau is read off these (e.g. the 99th
percentile of b on clean anatomy = "suppress the residual wherever the net is less sure than it ever is on healthy tissue"). The
pathological draw's b inside / outside its synthetic lesion is printed as a diagnostic only. Run on a GPU host that sees the curated
label maps, with the trainer's own flags:
  python scoring/sr_unc_tau.py --ckpt <super_resolve_200000.pth> --n 100 --out <json> -- --super_resolve --healthy \
      --pathology_prob 0.0 --attention --attention_identity_init --dark_csf 0.5 --bn_mode batch --amp off --gen_device cuda:0 \
      --data_dir <curated> --out_dir /tmp --seed 7 --strip_prob=0.5 --bspline_zoom --sr_uncertainty
(everything after -- is train_downstream's own argument list, the CMD line of the arm's log minus --steps/--tag/--init_from/--prefetch)."""
import sys, os, json, argparse, numpy as np, torch
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(here))
import train_downstream as T

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=100)
ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
mine, rest = ap.parse_known_args(); rest = [r for r in rest if r != "--"]
args = T.parse_args(rest)
torch.manual_seed(args.seed); np.random.seed(args.seed)
device = torch.device(mine.device); gen_device = torch.device(args.gen_device) if args.gen_device else device
T.pathology_primitives.DISABLED_MODES = {"Global Pathology", "Geometric Atrophy"}
ck = torch.load(mine.ckpt, map_location="cpu", weights_only=False); sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
assert "sr_uncertainty_marker" in sd, "not an --sr_uncertainty checkpoint"
model, _ = T.build_model(args, None, device, print, arch_state=sd); model.load_state_dict(sd); model.eval()
T.set_dark_csf(args.dark_csf, print); gen = T.build_generator(args, gen_device)
qs = [50, 90, 95, 99, 99.9]; clean_b, les_b, rest_b, clean_cases = [], [], [], []
with T.bn_eval(model, args.bn_mode), torch.no_grad():
    for i in range(mine.n):
        a = T.case_tensors(next(gen), device)
        for use_path in (False, True):
            lr = a["image_path"] if use_path else a["image_clean"]
            lr_n, _ = T.normalize(lr)
            res = model(lr_n[None, None])
            b = torch.exp(res[0, 1].float().clamp(-7.0, 2.0))
            brain = T.brain_mask(a["seg_path_idx"] if use_path else a["seg_clean_idx"])
            if not use_path:
                v = b[brain].flatten(); clean_b.append(v[torch.randperm(v.numel(), device=v.device)[:200000]].cpu().numpy())
                clean_cases.append([float(np.percentile(v.cpu().numpy(), q)) for q in qs])
            else:
                les = (a["seg_path_idx"] == T.LESION_IDX) & brain
                if bool(les.any()): les_b.append(float(b[les].mean())); rest_b.append(float(b[brain & ~les].mean()))
        if (i + 1) % 10 == 0: print(f"{i + 1}/{mine.n} draws", flush=True)
allb = np.concatenate(clean_b); pooled = {f"p{q}": float(np.percentile(allb, q)) for q in qs}
per_case = np.array(clean_cases); out = {"ckpt": mine.ckpt, "n_draws": mine.n, "clean_b_pooled_quantiles": pooled,
    "clean_b_per_case_quantile_median": {f"p{q}": float(np.median(per_case[:, j])) for j, q in enumerate(qs)},
    "path_draw_b_mean_lesion": float(np.mean(les_b)) if les_b else None, "path_draw_b_mean_rest": float(np.mean(rest_b)) if rest_b else None,
    "n_lesion_draws": len(les_b)}
json.dump(out, open(mine.out, "w"), indent=1); print(json.dumps(out, indent=1))
