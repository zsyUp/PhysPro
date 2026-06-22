
import glob
import os

import torch


def save_checkpoint(
    model,
    optimizer,
    scaler,
    n_iter,
    best_metrics,
    cfg,
    filename="latest_checkpoint.pth",
    is_best=False,
):
    """Save the full training state to disk."""
    checkpoint_dir = cfg.work_dir.ckpt_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, filename)
    checkpoint = {
        "iter": n_iter,
        "epoch": (n_iter // model.iters_per_epoch) + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_metrics": best_metrics,
        "cfg": cfg,
    }

    torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=True)
    print(f"Checkpoint saved: {checkpoint_path}")

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_hdmoe.pth")
        torch.save(checkpoint, best_path, _use_new_zipfile_serialization=True)
        print(f"Best model saved: {best_path}")

    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, device="cuda"):
    """Load model state and optional optimizer/scaler state from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, {"iter": 0, "miou": 0.0, "all_acc": 0.0, "avg_acc": 0.0}

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print("  Model state loaded")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("  Optimizer state loaded")

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print("  Scaler state loaded")

    start_iter = checkpoint.get("iter", 0)
    best_metrics = checkpoint.get(
        "best_metrics",
        {"iter": 0, "miou": 0.0, "all_acc": 0.0, "avg_acc": 0.0},
    )

    epoch = checkpoint.get("epoch", (start_iter // model.iters_per_epoch) + 1)
    print(f"  Resume from epoch {epoch}, iter {start_iter}")
    print(
        "  Best metrics: "
        f"mIoU={best_metrics['miou']:.4f}, "
        f"AllAcc={best_metrics['all_acc']:.4f}, "
        f"AvgAcc={best_metrics['avg_acc']:.4f}"
    )
    return start_iter, best_metrics


def find_latest_checkpoint(ckpt_dir):
    """Return the newest checkpoint path in a checkpoint directory."""
    checkpoint_pattern = os.path.join(ckpt_dir, "checkpoint_iter_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)

    latest_file = os.path.join(ckpt_dir, "latest_checkpoint.pth")
    if os.path.exists(latest_file):
        return latest_file

    if not checkpoint_files:
        return None

    checkpoint_iters = []
    for checkpoint_file in checkpoint_files:
        try:
            iter_num = int(checkpoint_file.split("_iter_")[-1].split(".pth")[0])
            checkpoint_iters.append((iter_num, checkpoint_file))
        except ValueError:
            continue

    if not checkpoint_iters:
        return None

    checkpoint_iters.sort(reverse=True)
    return checkpoint_iters[0][1]


def auto_resume_or_start_new(cfg, model, optimizer, scaler, device, resume_path=None):
    """Resume from a requested or latest checkpoint, otherwise start fresh."""
    if resume_path is not None:
        if os.path.exists(resume_path):
            print("\n" + "=" * 80)
            print(f"RESUMING FROM SPECIFIED CHECKPOINT: {resume_path}")
            print("=" * 80)
            start_iter, best_metrics = load_checkpoint(
                resume_path, model, optimizer, scaler, device
            )
            return start_iter, best_metrics, True

        print(f"\nSpecified checkpoint not found: {resume_path}")
        print("Searching for the latest checkpoint in the checkpoint directory...")

    ckpt_dir = cfg.work_dir.ckpt_dir
    latest_ckpt = find_latest_checkpoint(ckpt_dir)

    if latest_ckpt is not None:
        print("\n" + "=" * 80)
        print("RESUMING FROM LATEST CHECKPOINT")
        print("=" * 80)
        start_iter, best_metrics = load_checkpoint(
            latest_ckpt, model, optimizer, scaler, device
        )
        return start_iter, best_metrics, True

    print("\n" + "=" * 80)
    print("STARTING NEW TRAINING")
    print("=" * 80)
    return 0, {"iter": 0, "miou": 0.0, "all_acc": 0.0, "avg_acc": 0.0}, False


def save_periodic_checkpoint(
    model,
    optimizer,
    scaler,
    n_iter,
    best_metrics,
    cfg,
    save_interval=1000,
):
    """Save rolling checkpoints at a fixed iteration interval."""
    if (n_iter + 1) % save_interval == 0:
        filename = f"checkpoint_iter_{n_iter + 1}.pth"
        save_checkpoint(model, optimizer, scaler, n_iter, best_metrics, cfg, filename)
        save_checkpoint(
            model,
            optimizer,
            scaler,
            n_iter,
            best_metrics,
            cfg,
            "latest_checkpoint.pth",
        )


def cleanup_old_checkpoints(ckpt_dir, keep_last_n=3):
    """Delete older iteration checkpoints and keep only the newest few."""
    checkpoint_pattern = os.path.join(ckpt_dir, "checkpoint_iter_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if len(checkpoint_files) <= keep_last_n:
        return

    checkpoint_iters = []
    for checkpoint_file in checkpoint_files:
        try:
            iter_num = int(checkpoint_file.split("_iter_")[-1].split(".pth")[0])
            checkpoint_iters.append((iter_num, checkpoint_file))
        except ValueError:
            continue

    checkpoint_iters.sort(reverse=True)

    for _, old_file in checkpoint_iters[keep_last_n:]:
        try:
            os.remove(old_file)
            print(f"Removed old checkpoint: {old_file}")
        except OSError as exc:
            print(f"Failed to remove {old_file}: {exc}")
