"""
Shared helpers for train.py, test.py, infer.py, benchmark.py and dataset.py.

Only code used by more than one file lives here: config loading, seeding, device choice,
model building / checkpoint loading, evaluation, metrics, and the split-CSV format.
"""

import csv
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn

from tsm_mobilenetv3 import TSMMobileNetV3


# every key config.yaml must contain (missing keys fail loudly in load_config)
REQUIRED_KEYS = (
    "raw_dir", "frames_dir", "splits_dir", "classes", "frames_per_video", "frame_short_side",
    "val_ratio", "test_ratio",
    "arch", "n_segment", "shift_div", "causal_shift", "dropout", "pretrained", "image_size",
    "seed", "batch_size", "num_workers", "epochs", "lr", "momentum", "weight_decay", "amp",
    "checkpoint_dir", "results_dir",
    "infer_clip_seconds",
)

# frame files inside each video folder: 000000.jpg, 000001.jpg, ...
FRAME_NAME_FORMAT = "{:06d}.jpg"

# columns of train.csv / val.csv / test.csv
SPLIT_CSV_HEADER = ("video_id", "label", "num_frames")


# ---------------------------------------------------------------------------
# Config, seed, device
# ---------------------------------------------------------------------------
def load_config(path):
    """Read config.yaml and check that every required key is present.

    Args:
        path (str or Path): Path to the YAML config.

    Returns:
        dict: The config, keys as in config.yaml.

    Raises:
        KeyError: If a required key is missing.
        ValueError: If the YAML root is not a mapping or fewer than 2 classes are given.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: YAML root must be a mapping, got {type(cfg).__name__}")
    missing = [key for key in REQUIRED_KEYS if key not in cfg]
    if missing:
        raise KeyError(f"{path}: missing config keys {missing}")
    if len(cfg["classes"]) < 2:
        raise ValueError(f"{path}: need at least 2 classes, got {cfg['classes']}")
    return cfg


def set_seed(seed):
    """Seed python `random`, numpy and torch (CPU and all GPUs).

    Args:
        seed (int): The seed.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Pick the GPU if there is one, else the CPU.

    Returns:
        torch.device: "cuda" or "cpu".
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model and checkpoints
# ---------------------------------------------------------------------------
def build_model(cfg):
    """Build TSMMobileNetV3 from the model keys of the config.

    Args:
        cfg (dict): Config with keys classes, n_segment, arch, shift_div, causal_shift,
            dropout, pretrained.

    Returns:
        TSMMobileNetV3: The model (on CPU).
    """
    return TSMMobileNetV3(num_classes=len(cfg["classes"]), n_segment=cfg["n_segment"],
                          arch=cfg["arch"], shift_div=cfg["shift_div"],
                          causal_shift=cfg["causal_shift"], dropout=cfg["dropout"],
                          pretrained=cfg["pretrained"])


def load_checkpoint(path, device):
    """Rebuild the model saved by train.py and load its weights.

    The model is built from the config stored inside the checkpoint, so it always
    matches the trained architecture, whatever config.yaml says now.

    Args:
        path (str or Path): Checkpoint file (e.g. checkpoints/best.pt).
        device (torch.device): Device to put the model on.

    Returns:
        tuple:
            model (TSMMobileNetV3): Model in eval mode, on `device`.
            ckpt (dict): The whole checkpoint (epoch, model, optimizer, config, val_metrics).

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    # pretrained=False: the ImageNet weights would be overwritten anyway, so don't download them
    model = build_model(dict(ckpt["config"], pretrained=False))
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval(), ckpt


# ---------------------------------------------------------------------------
# Evaluation and metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run the model over a whole loader and collect predictions.

    Args:
        model (nn.Module): Video classifier, input (B, T, 3, H, W) -> logits (B, K).
        loader (DataLoader): Yields (clips (B, T, 3, H, W) float, labels (B,) long).
        criterion (nn.Module): Loss, e.g. nn.CrossEntropyLoss().
        device (torch.device): Device the model is on.

    Returns:
        tuple:
            mean_loss (float): Loss averaged over all clips.
            preds (Tensor): (N,), long, predicted class per clip (on CPU).
            labels (Tensor): (N,), long, true class per clip (on CPU).
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        logits = model(clips)                                  # (B, K)
        total_loss += criterion(logits, labels).item() * labels.shape[0]
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())
    preds = torch.cat(all_preds)     # (N,)
    labels = torch.cat(all_labels)   # (N,)
    return total_loss / labels.shape[0], preds, labels


def compute_metrics(preds, labels, num_classes, positive_class=1):
    """Accuracy, confusion matrix, and precision / recall / F1 of one class.

    Args:
        preds (Tensor): (N,), long, predicted classes in [0, num_classes).
        labels (Tensor): (N,), long, true classes in [0, num_classes).
        num_classes (int): K.
        positive_class (int): Class that precision / recall / F1 are reported for.
            Default 1 (the second entry of `classes`, e.g. "Fight").

    Returns:
        dict: accuracy, precision, recall, f1 (floats in [0, 1]) and
            confusion_matrix (K x K list of ints, rows = true class, columns = predicted).
            Precision / recall are 0.0 when their denominator is 0.

    Raises:
        AssertionError: If preds and labels have different shapes or are empty.
    """
    assert preds.shape == labels.shape, f"preds {tuple(preds.shape)} vs labels {tuple(labels.shape)}"
    assert preds.numel() > 0, "no predictions to score"

    # confusion[i, j] = number of clips of true class i predicted as class j
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    confusion.index_put_((labels, preds), torch.ones_like(labels), accumulate=True)

    p = positive_class
    true_pos = confusion[p, p].item()
    predicted_pos = confusion[:, p].sum().item()
    actual_pos = confusion[p, :].sum().item()
    precision = true_pos / predicted_pos if predicted_pos > 0 else 0.0
    recall = true_pos / actual_pos if actual_pos > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "accuracy": confusion.trace().item() / labels.numel(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion.tolist(),
    }


# ---------------------------------------------------------------------------
# Split CSVs and a fake dataset for self-tests
# ---------------------------------------------------------------------------
def write_split_csv(path, rows):
    """Write one split file (train.csv / val.csv / test.csv).

    Args:
        path (str or Path): Output CSV path (parent folders are created).
        rows (list of tuple): (video_id (str), label (int), num_frames (int)) per video;
            video_id is the video's frame folder relative to frames_dir, with "/" separators.

    Returns:
        None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SPLIT_CSV_HEADER)
        writer.writerows(rows)


def make_dummy_dataset(root, num_classes=2, videos_per_class=2, num_frames=8,
                       height=72, width=80, seed=0):
    """Write a tiny fake frames dataset + split CSVs, for the self-tests of other files.

    Every split (train / val / test) gets `videos_per_class` videos per class.
    Frames are random noise whose brightness depends on the class.

    Args:
        root (str or Path): Folder to write into (e.g. a temporary directory).
        num_classes (int): Number of classes. Default 2.
        videos_per_class (int): Videos per class per split. Default 2.
        num_frames (int): Frames per video. Default 8.
        height (int): Frame height in pixels. Default 72.
        width (int): Frame width in pixels. Default 80.
        seed (int): Seed for the noise. Default 0.

    Returns:
        tuple:
            frames_dir (Path): root/frames, one folder of JPGs per video.
            splits_dir (Path): root/splits, holding train.csv, val.csv, test.csv.
    """
    root = Path(root)
    frames_dir, splits_dir = root / "frames", root / "splits"
    gen = np.random.default_rng(seed)
    for split in ("train", "val", "test"):
        rows = []
        for label in range(num_classes):
            for k in range(videos_per_class):
                video_id = f"{split}/class{label}/video{k}"
                video_dir = frames_dir / video_id
                video_dir.mkdir(parents=True)
                base = 255 * (label + 1) / (num_classes + 1)   # class-dependent brightness
                for i in range(num_frames):
                    noise = gen.normal(0, 20, size=(height, width, 3))
                    pixels = np.clip(base + noise, 0, 255).astype(np.uint8)   # (H, W, 3)
                    Image.fromarray(pixels).save(video_dir / FRAME_NAME_FORMAT.format(i))
                rows.append((video_id, label, num_frames))
        write_split_csv(splits_dir / f"{split}.csv", rows)
    return frames_dir, splits_dir


def write_dummy_video(path, num_frames, height, width, fps=25):
    """Write a small random-noise .avi video, for the self-tests of other files.

    Args:
        path (str or Path): Output .avi path (parent folders are created).
        num_frames (int): Number of frames.
        height (int): Frame height in pixels.
        width (int): Frame width in pixels.
        fps (int): Frame rate stored in the file. Default 25.

    Returns:
        None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    gen = np.random.default_rng(0)
    for _ in range(num_frames):
        writer.write(gen.integers(0, 256, size=(height, width, 3), dtype=np.uint8))   # BGR
    writer.release()


# ---------------------------------------------------------------------------
# Self-test:  python utils.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1) load_config: a full config loads, a missing key fails loudly
        full_cfg = {key: 0 for key in REQUIRED_KEYS}
        full_cfg["classes"] = ["NonFight", "Fight"]
        (tmp / "ok.yaml").write_text(yaml.safe_dump(full_cfg), encoding="utf-8")
        assert load_config(tmp / "ok.yaml") == full_cfg, "loaded config differs from written one"
        broken_cfg = dict(full_cfg)
        del broken_cfg["lr"]
        (tmp / "broken.yaml").write_text(yaml.safe_dump(broken_cfg), encoding="utf-8")
        try:
            load_config(tmp / "broken.yaml")
            raise RuntimeError("expected a KeyError for the missing 'lr' key")
        except KeyError:
            pass
        print("[1] load_config ... OK")

        # 2) set_seed: the same seed gives the same random numbers in random / numpy / torch
        set_seed(123)
        first = (random.random(), np.random.rand(), torch.rand(1).item())
        set_seed(123)
        second = (random.random(), np.random.rand(), torch.rand(1).item())
        assert first == second, f"set_seed is not reproducible: {first} vs {second}"
        print("[2] set_seed ... OK")

        # 3) get_device matches what is available
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert get_device().type == expected, f"expected {expected}, got {get_device()}"
        print("[3] get_device ... OK")

        # 4) compute_metrics on known values: cm = [[1, 1], [1, 2]]
        labels = torch.tensor([0, 0, 1, 1, 1])
        preds = torch.tensor([0, 1, 1, 1, 0])
        m = compute_metrics(preds, labels, num_classes=2)
        assert m["confusion_matrix"] == [[1, 1], [1, 2]], f"confusion wrong: {m['confusion_matrix']}"
        assert abs(m["accuracy"] - 0.6) < 1e-9, f"accuracy wrong: {m['accuracy']}"
        for key in ("precision", "recall", "f1"):
            assert abs(m[key] - 2 / 3) < 1e-9, f"{key} wrong: {m[key]}"
        print("[4] compute_metrics known values ... OK")

        # 5) write_split_csv + make_dummy_dataset write the expected files
        frames_dir, splits_dir = make_dummy_dataset(tmp / "dummy", videos_per_class=2, num_frames=3)
        for split in ("train", "val", "test"):
            lines = (splits_dir / f"{split}.csv").read_text(encoding="utf-8").splitlines()
            assert lines[0] == ",".join(SPLIT_CSV_HEADER), f"bad header: {lines[0]}"
            assert len(lines) == 1 + 2 * 2, f"{split}.csv: expected 4 rows, got {len(lines) - 1}"
        n_jpgs = len(list(frames_dir.rglob("*.jpg")))
        assert n_jpgs == 3 * 2 * 2 * 3, f"expected 36 frames, got {n_jpgs}"
        print("[5] write_split_csv + make_dummy_dataset ... OK")

        # 6) build_model + evaluate on random clips: shapes and loss are sane
        model_cfg = {"classes": ["a", "b"], "n_segment": 4, "arch": "small", "shift_div": 8,
                     "causal_shift": False, "dropout": 0.5, "pretrained": False}
        model = build_model(model_cfg)
        clips = torch.randn(6, 4, 3, 64, 64)    # (N, T, 3, H, W)
        targets = torch.tensor([0, 1, 0, 1, 0, 1])
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(clips, targets), batch_size=4)
        loss, preds, labels = evaluate(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
        assert preds.shape == (6,) and torch.equal(labels, targets), "evaluate output wrong"
        assert np.isfinite(loss) and loss > 0, f"loss should be finite and positive, got {loss}"
        print("[6] build_model + evaluate ... OK")

        # 7) load_checkpoint restores exactly the saved weights
        ckpt_path = tmp / "ckpt.pt"
        torch.save({"epoch": 1, "model": model.state_dict(), "config": model_cfg}, ckpt_path)
        restored, ckpt = load_checkpoint(ckpt_path, torch.device("cpu"))
        for key, value in model.state_dict().items():
            assert torch.equal(value, restored.state_dict()[key]), f"weight {key} differs"
        assert not restored.training, "restored model should be in eval mode"
        print("[7] load_checkpoint round trip ... OK")

        # 8) write_dummy_video: OpenCV reads back the right frame count, size and fps
        write_dummy_video(tmp / "v.avi", num_frames=10, height=40, width=60, fps=25)
        cap = cv2.VideoCapture(str(tmp / "v.avi"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        assert len(frames) == 10 and frames[0].shape == (40, 60, 3), (
            f"read {len(frames)} frames of shape {frames[0].shape if frames else None}")
        assert round(fps) == 25, f"expected 25 fps, got {fps}"
        print("[8] write_dummy_video ... OK")

    print("All 8 checks passed.")
