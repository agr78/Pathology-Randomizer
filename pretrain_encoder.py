"""Pretrain an encoder to map motion-corrupted volumes where a frozen teacher
maps the clean ones.

The existing --motion_consistency term asks the encoder to produce the same
bottleneck for corrupted and clean input, with the clean side detached. That
works only because the segmentation loss runs alongside: the two encodings are
produced by the SAME weights, so mapping every input to a constant satisfies
the objective perfectly, and only the segmentation term makes a constant
useless. Detaching the target stops gradient through it but does not stop the
weights that produce it from moving.

Pretraining alone therefore needs a target the student cannot move. Here that
is a FROZEN teacher: a checkpoint already trained on clean data, held in eval
mode. The student sees the corrupted volume, the teacher sees the clean one,
and the student learns to land where the teacher does. A constant output would
score terribly against a teacher whose representation actually varies, so
collapse is not a solution.

This is distillation rather than self-distillation, and the target is stronger
than the current one -- the teacher's representation is good enough for ~0.78
anatomy Dice, so it is known to be informative rather than merely
self-consistent.

Matching is done at every encoder level, not just the bottleneck. Deep-only
matching leaves the early layers unconstrained and is less stable, and the
early layers are where motion artifact actually enters.

Usage:
    python pretrain_encoder.py --teacher .../synthseg_500000.pth --steps 50000
    python pretrain_encoder.py --teacher ... --from_teacher --steps 50000
"""
import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from crdmw_generator import paired_pathology_generator
from kspace_ops import simulate_kspace_motion
from networks import SynthSegUNet, load_checkpoint
from prefetch import Prefetcher
from util import make_mode_sampler

LABEL_LIST_SEGMENTATION = [0, 11, 12, 13, 16, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                           43, 44, 46, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 17, 47, 49, 51,
                           53, 55, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 48, 50,
                           52, 54, 56]


def encoder_features(model, x):
    """Every encoder level plus the bottleneck.

    Mirrors SynthSegUNet.forward's encoder path. Kept here rather than added to
    the network because this is a training-script concern and forward() already
    has a return_features flag for the bottleneck alone.
    """
    e1 = model.enc1(x)
    e2 = model.enc2(F.max_pool3d(e1, 2))
    e3 = model.enc3(F.max_pool3d(e2, 2))
    e4 = model.enc4(F.max_pool3d(e3, 2))
    b = model.bottleneck(F.max_pool3d(e4, 2))
    return [e1, e2, e3, e4, b]


def match_loss(student_feats, teacher_feats, mode="cosine"):
    """Distance between student and teacher features, summed over levels.

    Cosine per spatial position, as the existing consistency term does:
    bottleneck activations have large and drifting magnitudes (running_var
    there is ~1e3), so a raw MSE would be dominated by whichever level happens
    to have the largest scale and its weight would need retuning as training
    progresses. Cosine is scale-free and bounded.

    Levels are averaged rather than summed so the value stays in [0, 2]
    regardless of depth, which keeps it readable across configurations.
    """
    total = 0.0
    for s, t in zip(student_feats, teacher_feats):
        if mode == "mse":
            total = total + F.mse_loss(s, t.detach())
        else:
            cos = F.cosine_similarity(s.flatten(2), t.detach().flatten(2), dim=1)
            total = total + (1.0 - cos).mean()
    return total / len(student_feats)


def normalize_per_sample(x):
    """[0, 1] per sample, matching crdmw_train.py."""
    mins = x.amin(dim=(1, 2, 3, 4), keepdim=True)
    x = x - mins
    maxs = x.amax(dim=(1, 2, 3, 4), keepdim=True)
    return x / (maxs + 1e-7)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True,
                    help="Checkpoint whose encoder becomes the frozen target. Use a run trained "
                         "WITHOUT motion, so its representation is the clean one.")
    ap.add_argument("--data_dir",
                    default=os.environ.get("PR_DATA_DIR", "/path/to/brain_mris_QCed"))
    ap.add_argument("--out_dir",
                    default=os.environ.get("PR_OUT_DIR", "./runs"))
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--motion_prob", type=float, default=1.0,
                    help="Fraction of steps that corrupt the student's input. At 1.0 every step "
                         "is a corrupted/clean pair, which is the point of the pretraining; "
                         "below 1.0 some steps ask the student to match the teacher on the SAME "
                         "clean volume, which is a trivial target if --from_teacher is set.")
    ap.add_argument("--from_teacher", action="store_true",
                    help="Initialise the student from the teacher's weights. Converges faster "
                         "but starts near zero loss, so it may barely move -- worth comparing "
                         "against random initialisation rather than assuming.")
    ap.add_argument("--loss", default="cosine", choices=["cosine", "mse"])
    ap.add_argument("--healthy", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gen_device", default=None,
                    help="Run the generator on a separate CUDA device. Measured at 1.41x on two "
                         "A100s; see crdmw_train.py.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"encoder_pretrain_out", f"log_{stamp}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_file = open(os.path.join(run_dir, "train_log.txt"), "a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"Encoder pretraining -> {run_dir}")
    for k, v in sorted(vars(args).items()):
        log(f"    {k:20s} {v}")

    # Teacher: frozen, eval mode. Its BatchNorm running statistics are used as
    # stored rather than recomputed, which is what makes it a fixed target --
    # in train mode it would adapt to whatever the student is being shown.
    teacher, _, _ = load_checkpoint(args.teacher, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    log(f"teacher loaded, frozen: {args.teacher}")

    student = SynthSegUNet(out_channels=56).to(device)
    if args.from_teacher:
        student.load_state_dict(teacher.state_dict())
        log("student initialised FROM TEACHER (starts near zero loss)")
    else:
        log("student initialised randomly")

    # Only encoder parameters are trained; the decoder is untouched here and
    # gets trained afterwards by crdmw_train.py --resume from this checkpoint.
    enc_params = []
    for name in ("enc1", "enc2", "enc3", "enc4", "bottleneck"):
        enc_params += list(getattr(student, name).parameters())
    opt = torch.optim.Adam(enc_params, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    log(f"training {sum(p.numel() for p in enc_params)/1e6:.1f}M encoder parameters "
        f"of {sum(p.numel() for p in student.parameters())/1e6:.1f}M total")

    gen = paired_pathology_generator(
        args.data_dir, make_mode_sampler(restricted=False),
        label_list_segmentation=LABEL_LIST_SEGMENTATION,
        n_neutral_labels=20, label_list_normalization=[1, 18], sides="both",
        labels_to_kill=[45], list_to_kill_photo=[], list_to_kill_photo_inc_ventricles=[],
        flipping=True, mix_real_images=False,
        # No pathology: this stage learns motion invariance on healthy anatomy,
        # and generating the pathological stream would double the cost for
        # something the loss never looks at.
        generate_pathology=False, minimal_io=True, compact_labels=True,
        random_kill_labels=list(range(34, 47)), random_kill_prob=0.5,
        random_kill_frac=(1.0, 1.0), healthy_only=args.healthy,
        device=(args.gen_device or str(device)))
    gen = Prefetcher(gen, depth=2)

    t_start = time.time()
    running = []
    for i in range(1, args.steps + 1):
        case = next(gen)
        torch.set_grad_enabled(True)

        clean = case["image_clean"].detach().to(device).float()[None, None]

        # Corrupt a COPY: the teacher must see the uncorrupted volume, so the
        # two inputs differ only by the artifact being learned away.
        if np.random.rand() < args.motion_prob:
            corrupted = simulate_kspace_motion(
                clean.squeeze().clone(), device,
                num_segments=np.random.randint(4, 12))[None, None]
        else:
            corrupted = clean.clone()

        x_student = normalize_per_sample(corrupted)
        x_teacher = normalize_per_sample(clean)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                t_feats = encoder_features(teacher, x_teacher)
            s_feats = encoder_features(student, x_student)
            loss = match_loss(s_feats, t_feats, args.loss)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(enc_params, max_norm=1.0)
        scaler.step(opt)
        scaler.update()

        running.append(loss.item())
        if i % 100 == 0:
            # Per-level values as well as the mean: if the bottleneck matches
            # while enc1 does not, the student is agreeing at the depth where
            # everything looks similar and has not learned the artifact.
            with torch.no_grad():
                per_level = [
                    float((1.0 - F.cosine_similarity(s.flatten(2), t.flatten(2), dim=1)).mean())
                    for s, t in zip(s_feats, t_feats)]
            log(f"Step {i:06d} | loss {np.mean(running[-100:]):.4f} | "
                + " ".join(f"L{k+1} {v:.3f}" for k, v in enumerate(per_level))
                + f" | {(time.time()-t_start)/i:.2f}s/step")

        if i % 5000 == 0:
            path = os.path.join(ckpt_dir, f"encoder_{i:06d}.pth")
            torch.save({"model_state_dict": student.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "step": i, "teacher": args.teacher,
                        "note": "encoder pretrained by distillation; decoder is untrained "
                                "unless --from_teacher was used"}, path)
            log(f"saved {path}")

    log("PRETRAINING COMPLETE")
    log("Next: train the decoder from this checkpoint with crdmw_train.py. The "
        "saved state_dict has the full network shape, so it loads directly.")
    log_file.close()


if __name__ == "__main__":
    main()