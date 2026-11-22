"""
Train TSMMobileNetV3 on the frames prepared by prepare_data.py.

Every epoch: one pass over train, then validation. Saves checkpoint_dir/last.pt every
epoch and checkpoint_dir/best.pt whenever the validation accuracy improves.

Training recipe (SGD with momentum, cosine learning-rate schedule, ImageNet init) follows
the official TSM training scripts:
    https://github.com/mit-han-lab/temporal-shift-module (main.py, --lr_type cos)
    Paper: Lin, Gan, Han, "TSM: Temporal Shift Module for Efficient Video Understanding",
           ICCV 2019 (arXiv:1811.08383)

Usage:
    python train.py [--config config.yaml]
    python train.py --self-test
"""

import argparse
import math
import tempfile
from pathlib import Path

import torch  # type: ignore[import-not-found,import-untyped]
from torch import nn  # type: ignore[import-not-found,import-untyped]
from torch.utils.data import DataLoader  # type: ignore[import-not-found,import-untyped]

from dataset import VideoDataset
from utils import (build_model, compute_metrics, evaluate, get_device, load_checkpoint,
                   load_config, make_dummy_dataset, set_seed)


def make_loaders(cfg, device):
    """Build the train and val DataLoaders from the split CSVs.

    Args:
        cfg (dict): Config (splits_dir, frames_dir, n_segment, image_size, batch_size,
            num_workers).
        device (torch.device): Training device (pinned memory is used on a GPU).

    Returns:
        tuple:
            train_loader (DataLoader): Shuffled, augmented, drops the last partial batch.
            val_loader (DataLoader): Fixed order, no augmentation.
        Both yield (clips (B, T, 3, image_size, image_size) float, labels (B,) long).
    """
    splits_dir = Path(cfg["splits_dir"])
    loaders = []
    for split, is_train in (("train", True), ("val", False)):
        ds = VideoDataset(splits_dir / f"{split}.csv", cfg["frames_dir"], cfg["n_segment"],
                          cfg["image_size"], train=is_train)
        # drop_last in training: a batch of 1 clip breaks BatchNorm's batch statistics
        loaders.append(DataLoader(ds, batch_size=cfg["batch_size"], shuffle=is_train,
                                  num_workers=cfg["num_workers"],
                                  pin_memory=device.type == "cuda", drop_last=is_train))
    return tuple(loaders)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    """Run one training pass over the loader.

    Args:
        model (nn.Module): Video classifier, (B, T, 3, H, W) -> logits (B, K).
        loader (DataLoader): Yields (clips (B, T, 3, H, W) float, labels (B,) long).
        optimizer (torch.optim.Optimizer): Optimizer over the model's parameters.
        criterion (nn.Module): Loss, e.g. nn.CrossEntropyLoss().
        device (torch.device): Device the model is on.
        scaler (torch.amp.GradScaler): Loss scaler for mixed precision (disabled = no-op).
        use_amp (bool): Run the forward pass under autocast (mixed precision).

    Returns:
        tuple:
            mean_loss (float): Training loss averaged over all clips.
            accuracy (float): Training accuracy in [0, 1].
    """
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(clips)                   # (B, K)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.shape[0]
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.shape[0]
    assert seen > 0, "the train loader gave no batches (fewer clips than batch_size?)"
    return total_loss / seen, correct / seen


def train(cfg):
    """Train the model for cfg["epochs"] epochs and save the checkpoints.

    Args:
        cfg (dict): The full config (see config.yaml).

    Returns:
        list of dict: One entry per epoch: epoch, lr, train_loss, train_acc, val_loss
            and the val metrics from utils.compute_metrics (val_accuracy, val_f1, ...).
    """
    set_seed(cfg["seed"])
    device = get_device()
    use_amp = cfg["amp"] and device.type == "cuda"   # autocast float16 only helps on a GPU
    print(f"device: {device}, mixed precision: {use_amp}")

    train_loader, val_loader = make_loaders(cfg, device)
    print(f"videos: train {len(train_loader.dataset)}, val {len(val_loader.dataset)}")

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                                weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion,
                                                device, scaler, use_amp)
        val_loss, preds, labels = evaluate(model, val_loader, criterion, device)
        val_metrics = compute_metrics(preds, labels, num_classes=len(cfg["classes"]))
        scheduler.step()

        print(f"epoch {epoch}/{cfg['epochs']}  lr={lr:.5f}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
              f"val_f1={val_metrics['f1']:.4f}")
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                        "train_acc": train_acc, "val_loss": val_loss,
                        **{f"val_{key}": value for key, value in val_metrics.items()}})

        # save last.pt every epoch, best.pt when the val accuracy improves
        state = {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "config": cfg, "val_metrics": val_metrics}
        torch.save(state, ckpt_dir / "last.pt")
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(state, ckpt_dir / "best.pt")
            print(f"  new best val_acc={best_acc:.4f}, saved {ckpt_dir / 'best.pt'}")

    print(f"done. best val_acc={best_acc:.4f}")
    return history


def _self_test():
    """Check the training pieces end to end on a tiny fake dataset (CPU, a few seconds).

    Returns:
        None
    """
    set_seed(0)
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        frames_dir, splits_dir = make_dummy_dataset(tmp, videos_per_class=3, num_frames=8)
        # str paths: the config is stored in checkpoints, and torch.load(weights_only=True)
        # only accepts plain python types
        cfg = {"frames_dir": str(frames_dir), "splits_dir": str(splits_dir),
               "checkpoint_dir": str(Path(tmp) / "ckpt"), "classes": ["NonFight", "Fight"],
               "arch": "small", "n_segment": 4, "shift_div": 8, "causal_shift": False,
               "dropout": 0.5, "pretrained": False, "image_size": 64, "seed": 0,
               "batch_size": 2, "num_workers": 0, "epochs": 2, "lr": 0.01, "momentum": 0.9,
               "weight_decay": 0.0005, "amp": False}

        # 1) make_loaders: 6 train clips / batch 2 = 3 batches, 6 val clips, model input shape
        train_loader, val_loader = make_loaders(cfg, device)
        assert len(train_loader) == 3, f"expected 3 train batches, got {len(train_loader)}"
        assert len(val_loader.dataset) == 6, f"expected 6 val clips, got {len(val_loader.dataset)}"
        clips, labels = next(iter(train_loader))
        assert clips.shape == (2, 4, 3, 64, 64), f"bad batch shape {tuple(clips.shape)}"
        print("[1] make_loaders ... OK")

        # 2) train_one_epoch: finite loss, accuracy in [0, 1], weights change
        model = build_model(cfg)
        before = [p.detach().clone() for p in model.parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        loss, acc = train_one_epoch(model, train_loader, optimizer, nn.CrossEntropyLoss(),
                                    device, scaler, use_amp=False)
        assert math.isfinite(loss) and 0 <= acc <= 1, f"bad loss/acc: {loss}, {acc}"
        changed = sum(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
        assert changed > 0, "no parameter changed after a training epoch"
        print("[2] train_one_epoch ... OK")

        # 3) train: history per epoch, cosine lr goes down, last.pt / best.pt saved correctly
        history = train(cfg)
        assert [h["epoch"] for h in history] == [1, 2], f"bad history epochs {history}"
        assert history[1]["lr"] < history[0]["lr"], "cosine schedule should lower the lr"
        best_epoch = max(history, key=lambda h: h["val_accuracy"])["epoch"]   # first max
        last, best = (torch.load(Path(cfg["checkpoint_dir"]) / name, weights_only=True)
                      for name in ("last.pt", "best.pt"))
        assert last["epoch"] == 2, f"last.pt should be epoch 2, got {last['epoch']}"
        assert best["epoch"] == best_epoch, f"best.pt epoch {best['epoch']}, expected {best_epoch}"
        print("[3] train + checkpoints ... OK")

        # 4) re-loading best.pt reproduces the val accuracy recorded during training
        model, ckpt = load_checkpoint(Path(cfg["checkpoint_dir"]) / "best.pt", device)
        _, preds, labels = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
        acc = compute_metrics(preds, labels, num_classes=2)["accuracy"]
        assert acc == ckpt["val_metrics"]["accuracy"], (
            f"reloaded val acc {acc} != saved {ckpt['val_metrics']['accuracy']}")
        print("[4] best.pt reproduces its val accuracy ... OK")

    print("All 4 checks passed.")


def main():
    """Parse the command line and train (or run the self-test).

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Train TSM-MobileNetV3")
    parser.add_argument("--config", default="config.yaml", help="path to the YAML config")
    parser.add_argument("--self-test", action="store_true", help="run the self-test and exit")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        train(load_config(args.config))


if __name__ == "__main__":
    main()
