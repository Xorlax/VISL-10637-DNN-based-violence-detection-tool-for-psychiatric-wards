"""
Evaluate a trained checkpoint on the test split (or any split).

Reports accuracy, precision / recall / F1 of the positive class (classes[1], e.g. Fight)
and the confusion matrix; prints them and saves them to results_dir/<split>_metrics.json.

Usage:
    python test.py [--config config.yaml] [--checkpoint checkpoints/best.pt] [--split test]
    python test.py --self-test
"""

import argparse
import json
import tempfile
from pathlib import Path

import torch  # type: ignore[import-not-found,import-untyped]
from torch import nn  # type: ignore[import-not-found,import-untyped]
from torch.utils.data import DataLoader  # type: ignore[import-not-found,import-untyped]

from dataset import VideoDataset
from utils import (build_model, compute_metrics, evaluate, get_device, load_checkpoint,
                   load_config, make_dummy_dataset, set_seed)


def run_test(cfg, checkpoint_path, split):
    """Evaluate a checkpoint on one split and save the metrics as JSON.

    The model and the clip settings (n_segment, image_size) come from the config stored
    in the checkpoint, so they always match training; the data paths and batch settings
    come from `cfg`.

    Args:
        cfg (dict): Current config (frames_dir, splits_dir, batch_size, num_workers,
            results_dir).
        checkpoint_path (str or Path): Checkpoint written by train.py.
        split (str): "train", "val" or "test".

    Returns:
        dict: Metrics from utils.compute_metrics plus loss, split, checkpoint, epoch, classes.
    """
    device = get_device()
    model, ckpt = load_checkpoint(checkpoint_path, device)
    train_cfg = ckpt["config"]

    ds = VideoDataset(Path(cfg["splits_dir"]) / f"{split}.csv", cfg["frames_dir"],
                      train_cfg["n_segment"], train_cfg["image_size"], train=False)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
    loss, preds, labels = evaluate(model, loader, nn.CrossEntropyLoss(), device)

    metrics = compute_metrics(preds, labels, num_classes=len(train_cfg["classes"]))
    metrics.update({"loss": loss, "split": split, "checkpoint": str(checkpoint_path),
                    "epoch": ckpt["epoch"], "classes": train_cfg["classes"]})

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{split}_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(format_report(metrics))
    print(f"saved {out_path}")
    return metrics


def format_report(metrics):
    """Turn the metrics into a readable text report (scores + labelled confusion matrix).

    Args:
        metrics (dict): Output of run_test (needs accuracy, precision, recall, f1, loss,
            confusion_matrix, classes, split, epoch).

    Returns:
        str: Multi-line report.
    """
    classes = metrics["classes"]
    width = max(len(name) for name in classes) + 2
    lines = [f"{metrics['split']} set, checkpoint epoch {metrics['epoch']}, "
             f"{sum(map(sum, metrics['confusion_matrix']))} clips",
             f"  loss      {metrics['loss']:.4f}",
             f"  accuracy  {metrics['accuracy']:.4f}",
             f"  precision {metrics['precision']:.4f}   ({classes[1]})",
             f"  recall    {metrics['recall']:.4f}   ({classes[1]})",
             f"  f1        {metrics['f1']:.4f}   ({classes[1]})",
             "  confusion matrix (rows = true, columns = predicted):",
             "  " + " " * width + "".join(f"{name:>{width}}" for name in classes)]
    for name, row in zip(classes, metrics["confusion_matrix"]):
        lines.append("  " + f"{name:<{width}}" + "".join(f"{n:>{width}}" for n in row))
    return "\n".join(lines)


def _self_test():
    """Check run_test and format_report on a tiny fake dataset with an untrained model.

    Returns:
        None
    """
    set_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frames_dir, splits_dir = make_dummy_dataset(tmp, videos_per_class=3, num_frames=8)
        train_cfg = {"classes": ["NonFight", "Fight"], "arch": "small", "n_segment": 4,
                     "shift_div": 8, "causal_shift": False, "dropout": 0.5,
                     "pretrained": False, "image_size": 64}
        ckpt_path = tmp / "best.pt"
        torch.save({"epoch": 3, "model": build_model(train_cfg).state_dict(),
                    "config": train_cfg}, ckpt_path)
        # n_segment / image_size differ on purpose: run_test must use the checkpoint's values
        cfg = {"frames_dir": frames_dir, "splits_dir": splits_dir, "results_dir": tmp / "res",
               "batch_size": 4, "num_workers": 0, "n_segment": 99, "image_size": 999}

        # 1) metrics are consistent with the 6 test clips and the checkpoint's settings
        metrics = run_test(cfg, ckpt_path, "test")
        cm = metrics["confusion_matrix"]
        assert sum(map(sum, cm)) == 6, f"confusion matrix should count 6 clips: {cm}"
        assert [sum(row) for row in cm] == [3, 3], f"each class has 3 test clips: {cm}"
        assert abs(metrics["accuracy"] - (cm[0][0] + cm[1][1]) / 6) < 1e-9, "accuracy != trace/N"
        assert metrics["epoch"] == 3 and metrics["split"] == "test", "wrong epoch/split"
        print("[1] run_test metrics ... OK")

        # 2) the JSON file holds exactly the returned metrics
        saved = json.loads((tmp / "res" / "test_metrics.json").read_text(encoding="utf-8"))
        assert saved == metrics, "saved JSON differs from returned metrics"
        print("[2] metrics JSON saved ... OK")

        # 3) format_report shows the scores and a labelled confusion matrix
        report = format_report(metrics)
        assert f"accuracy  {metrics['accuracy']:.4f}" in report, "accuracy missing in report"
        assert report.count("NonFight") == 2 and "(Fight)" in report, "class names missing"
        print("[3] format_report ... OK")

        # 4) a missing checkpoint fails loudly
        try:
            run_test(cfg, tmp / "missing.pt", "test")
            raise RuntimeError("expected FileNotFoundError for a missing checkpoint")
        except FileNotFoundError:
            pass
        print("[4] missing checkpoint rejected ... OK")

    print("All 4 checks passed.")


def main():
    """Parse the command line and evaluate (or run the self-test).

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Evaluate a TSM-MobileNetV3 checkpoint")
    parser.add_argument("--config", default="config.yaml", help="path to the YAML config")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint to evaluate (default: <checkpoint_dir>/best.pt)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--self-test", action="store_true", help="run the self-test and exit")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    cfg = load_config(args.config)
    checkpoint = args.checkpoint or Path(cfg["checkpoint_dir"]) / "best.pt"
    run_test(cfg, checkpoint, args.split)


if __name__ == "__main__":
    main()
