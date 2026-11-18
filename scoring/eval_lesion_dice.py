"""Evaluate the predicted Lesion label against BraTS ground-truth tumour masks.

compare_bn_cases.py measures eval/train self-consistency, which says nothing
about whether a segmentation is correct -- a model can be perfectly
self-consistent and wrong everywhere. This compares against the manual
segmentations.

BraTS labels: 1 = necrotic core, 2 = peritumoural edema, 3 = enhancing tumour.
The network predicts a single Lesion class (compact index 7, FreeSurfer id 33),
so the ground truth has to be merged. Which labels are merged changes the
answer substantially, hence --target:

    whole  (1,2,3)  the full abnormality including edema  [default]
    core   (1,3)    necrotic + enhancing, excludes edema
    enhancing (3)   enhancing tumour only

Whole tumour is the fairest default: the pathology generator's lesions are
intensity abnormalities without a necrotic/enhancing distinction, so asking
the model to separate sub-compartments it was never trained on would
understate it.

Usage:
    python evaluate_lesion_dice.py \\
        --ckpt baseline=/path/synthseg_65000.pth \\
        --ckpt pathology=/path/synthseg_65000.pth \\
        --n_cases 10
"""
import argparse
import csv
import glob
import os
import random
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from compare_datasets import (find_t1_files, head_bbox, normalize_like_training,
                              to_ras, from_ras, LABEL_LIST_SEGMENTATION,
                              LESION_COMPACT_IDX, TARGET_LABELS, DATASETS,
                              dataset_key, list_cases, dice)
# Makes the networks.py import below resolve regardless of whether this
# file lives in the repo root or a subdirectory such as utils/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks import load_checkpoint

# LABEL_LIST_SEGMENTATION, LESION_COMPACT_IDX, TARGET_LABELS, DATASETS,
# dataset_key, list_cases, dice, to_ras, and from_ras are defined once in
# compare_datasets.py and imported above rather than duplicated here.



def predict_lesion(model, img_data, device, use_train_mode=False):
    """Segment and return the predicted Lesion mask on the original grid."""
    bbox = head_bbox(img_data)
    work = img_data[bbox]
    x = torch.from_numpy(np.ascontiguousarray(work)).to(device).unsqueeze(0).unsqueeze(0)
    x = normalize_like_training(x)
    pads = [(32 - s % 32) % 32 for s in x.shape[2:]]
    xp = F.pad(x, (0, pads[2], 0, pads[1], 0, pads[0]))

    model.train() if use_train_mode else model.eval()
    with torch.no_grad():
        out = model(xp)
        out = out[:, :, :x.shape[2], :x.shape[3], :x.shape[4]]
        pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

    full = np.zeros(img_data.shape, dtype=np.int32)
    full[bbox] = pred
    # Skull-stripped: the network was trained on intact heads and emits
    # arbitrary labels into the stripped void, so mask it out.
    full[img_data <= 0] = 0
    return full


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--dataset", default="BraTS2023-GLI")
    ap.add_argument("--data_root", default=os.environ.get('PR_EVAL_DATA_ROOT', '/path/to/eval_data_root'))
    ap.add_argument("--target", default="whole", choices=sorted(TARGET_LABELS))
    ap.add_argument("--modality", default=None, metavar="FILENAME",
                    help="Override which volume to segment, e.g. FLAIR.nii.gz for WMH. SynthSeg "
                         "is contrast-agnostic by construction (trained on random GMM contrasts), "
                         "and WMH lesions are bright on FLAIR but near-invisible on T1. The file "
                         "must be on the same grid as the ground-truth mask.")
    ap.add_argument("--n_cases", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", default=None, metavar="DIR",
                    help="Write predicted Lesion masks to DIR/<model_name>/<sub>_lesion.nii.gz, "
                         "plus the merged ground truth once to DIR/ground_truth/. Lets you "
                         "overlay them and check whether a near-zero Dice is a localisation "
                         "failure or a misalignment bug -- comparable predicted and ground-truth "
                         "volumes with ~0 overlap looks more like the latter.")
    ap.add_argument("--train_mode", action="store_true",
                    help="Segment in train mode instead of eval. BatchNorm then uses the "
                         "volume's own statistics rather than the stored ones -- not how the "
                         "model would be deployed, but it isolates what the weights know.")
    ap.add_argument("--save_scores", default=None, metavar="PATH",
                    help="Append every (dataset, model, subject, dice) row to this CSV. "
                         "Appends rather than overwrites, and writes the header only if the "
                         "file doesn't exist yet -- so BraTS/ATLAS/WMH runs (or repeated runs "
                         "across checkpoints) can all accumulate into one file. Pair with "
                         "show_top5.py to summarize it.")
    args = ap.parse_args()

    models = []
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        if not os.path.isfile(path):
            raise SystemExit(f"not found: {path}")
        models.append((name, path))

    data_dir = os.path.join(args.data_root, args.dataset)
    paired = list_cases(data_dir, args.dataset, modality=args.modality)
    if not paired:
        raise SystemExit(f"no T1/ground-truth pairs found under {data_dir}")

    random.seed(args.seed)
    picks = random.sample(paired, min(args.n_cases, len(paired)))
    key = dataset_key(args.dataset)
    wanted = DATASETS[key]["labels"] or TARGET_LABELS[args.target]
    print(f"{len(paired)} cases with ground truth; using {len(picks)} (seed {args.seed})")
    print(f"dataset kind = {key}, target labels {wanted}, "
          f"mode = {'train' if args.train_mode else 'eval'}\n")

    results = {name: [] for name, _ in models}

    csv_writer = None
    csv_file = None
    if args.save_scores:
        is_new = not os.path.isfile(args.save_scores)
        csv_file = open(args.save_scores, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if is_new:
            csv_writer.writerow(["dataset", "model", "subject", "dice", "pred_vox", "gt_vox"])

    for name, path in models:
        model, _, _ = load_checkpoint(path, device := torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"), verbose=False)
        print(f"--- {name} ---")
        for sub, t1, gtp in picks:
            t1_nii = nib.load(t1)
            img, inv_ornt = to_ras(t1_nii)
            img = img.astype(np.float32)
            gt_img = nib.load(gtp)
            gt, _ = to_ras(gt_img)
            gt = gt.astype(np.int32)
            if gt.shape != img.shape:
                print(f"  {sub}: SKIPPED, grid mismatch {gt.shape} vs {img.shape}")
                continue
            gt_mask = np.isin(gt, wanted)
            pred_labels = predict_lesion(model, img, device, use_train_mode=args.train_mode)
            pred = pred_labels == LESION_COMPACT_IDX
            d = dice(pred, gt_mask)

            # What DOES the model call the lesion region? A dark chronic-stroke
            # cavity read as CSF, or a WMH read as white matter, is a very
            # different failure from predicting nothing there at all.
            inside = pred_labels[gt_mask]
            if inside.size:
                vals, counts = np.unique(inside, return_counts=True)
                order = np.argsort(-counts)[:3]
                top = "  ".join(
                    f"{LABEL_LIST_SEGMENTATION[vals[i]]}:{100*counts[i]/inside.size:.0f}%"
                    if vals[i] < len(LABEL_LIST_SEGMENTATION) else f"?{vals[i]}"
                    for i in order)
            else:
                top = "-"
            results[name].append((sub, d, int(pred.sum()), int(gt_mask.sum())))
            if csv_writer:
                # d can be None (both masks empty) -- written as an empty
                # field rather than skipped, so show_top5.py's row count
                # still matches "cases attempted" and a silently-dropped
                # case doesn't look identical to one that was never run.
                csv_writer.writerow([args.dataset, name, sub,
                                     f"{d:.6f}" if d is not None else "",
                                     int(pred.sum()), int(gt_mask.sum())])
                csv_file.flush()

            if args.save_dir:
                # Reuse the T1's affine and header so the mask lands on the
                # same grid in a viewer -- writing an identity affine would
                # make a correct prediction look misaligned.
                ref = nib.load(t1)
                mdir = os.path.join(args.save_dir, name)
                os.makedirs(mdir, exist_ok=True)
                nib.save(nib.Nifti1Image(from_ras(pred, inv_ornt).astype(np.uint8),
                                         ref.affine, ref.header),
                         os.path.join(mdir, f"{sub}_lesion.nii.gz"))
                gdir = os.path.join(args.save_dir, "ground_truth")
                os.makedirs(gdir, exist_ok=True)
                gpath = os.path.join(gdir, f"{sub}_{args.target}.nii.gz")
                if not os.path.isfile(gpath):
                    nib.save(nib.Nifti1Image(from_ras(gt_mask, inv_ornt).astype(np.uint8),
                                             ref.affine, ref.header),
                             gpath)
            ds = "  n/a" if d is None else f"{d:5.3f}"
            print(f"  {sub}  dice {ds}   pred {pred.sum():7d}  gt {gt_mask.sum():7d}   "
                  f"in-gt: {top}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print()

    if args.save_dir:
        print(f"masks written under {args.save_dir}")
        print("  overlay in freeview to check alignment:")
        print(f"    freeview <T1>.nii.gz \\")
        print(f"      {os.path.join(args.save_dir, 'ground_truth', '<sub>_' + args.target + '.nii.gz')}:colormap=jet:opacity=0.4 \\")
        for name, _ in models:
            print(f"      {os.path.join(args.save_dir, name, '<sub>_lesion.nii.gz')}:colormap=heat:opacity=0.4 \\")
        print()

    print("=" * 60)
    for name, _ in models:
        ds = np.array([r[1] for r in results[name] if r[1] is not None])
        if len(ds) == 0:
            print(f"{name:<14} no scorable cases")
            continue
        print(f"{name:<14} dice {ds.mean():.3f} ± {ds.std():.3f}  "
              f"[{ds.min():.3f}, {ds.max():.3f}]  n={len(ds)}")
    print("=" * 60)

    if len(models) == 2:
        a, b = models[0][0], models[1][0]
        da = {s: d for s, d, _, _ in results[a] if d is not None}
        db = {s: d for s, d, _, _ in results[b] if d is not None}
        common = sorted(set(da) & set(db))
        if common:
            diff = np.array([db[s] - da[s] for s in common])
            print(f"\npaired difference ({b} - {a}): {diff.mean():+.3f} ± {diff.std():.3f}")
            print(f"  {b} higher on {(diff > 0).sum()}/{len(diff)} cases")

    if csv_file:
        csv_file.close()
        print(f"\nscores appended to {args.save_scores}")


if __name__ == "__main__":
    main()