"""
Video clip dataset: reads the frames written by prepare_data.py and returns
(clip, label) pairs for TSMMobileNetV3.

Sources:
  * Segment-based frame sampling (train: random frame per segment, eval: center frame)
    as in the official TSM code, ops/dataset.py (_sample_indices / _get_val_indices)
      https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/dataset.py
    which follows Temporal Segment Networks:
      Paper: Wang et al., "Temporal Segment Networks: Towards Good Practices for Deep Action
             Recognition", ECCV 2016 (arXiv:1608.00859)

Clip shape convention: (T, 3, H, W), T = n_segment; a DataLoader batches it to (B, T, 3, H, W).
"""

import csv
import tempfile
from pathlib import Path

import numpy as np  # type: ignore[import-not-found,import-untyped]
import torch  # type: ignore[import-not-found,import-untyped]
from torch.utils.data import Dataset  # type: ignore[import-not-found,import-untyped]
from torchvision.io import ImageReadMode, read_image  # type: ignore[import-not-found,import-untyped]

from utils import FRAME_NAME_FORMAT, SPLIT_CSV_HEADER, make_dummy_dataset, set_seed


# ImageNet statistics, because the backbone is pretrained on ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FLIP_PROB = 0.5


def read_split_csv(path):
    """Read one split file written by prepare_data.py.

    Args:
        path (str or Path): train.csv, val.csv or test.csv.

    Returns:
        list of tuple: (video_id (str), label (int), num_frames (int)) per video.

    Raises:
        ValueError: If the header is wrong or the file has no videos.
    """
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = tuple(next(reader))
        if header != SPLIT_CSV_HEADER:
            raise ValueError(f"{path}: expected header {SPLIT_CSV_HEADER}, got {header}")
        rows = [(video_id, int(label), int(num_frames)) for video_id, label, num_frames in reader]
    if not rows:
        raise ValueError(f"{path}: no videos listed")
    return rows


def sample_indices(num_frames, n_segment, train):
    """Choose which T frames of a video form the clip (one frame per equal-length segment).

    Source: TSM's ops/dataset.py, _sample_indices (train) and _get_val_indices (eval).

    Args:
        num_frames (int): Frames available for the video (>= 1).
        n_segment (int): T, frames to pick.
        train (bool): True = random frame inside each segment (augmentation),
            False = the center frame of each segment (deterministic).

    Returns:
        list of int: T frame indices in [0, num_frames), non-decreasing.
            When num_frames < T some frames are repeated.

    Raises:
        AssertionError: If num_frames < 1.
    """
    assert num_frames >= 1, f"video has no frames (num_frames={num_frames})"
    segment_len = num_frames // n_segment
    if train and segment_len > 0:
        offsets = np.random.randint(segment_len, size=n_segment)
        return [int(i * segment_len + offsets[i]) for i in range(n_segment)]
    # center of each segment (also used in training when the video is shorter than T)
    tick = num_frames / n_segment
    return [int(tick / 2 + tick * i) for i in range(n_segment)]


def load_frames(video_dir, indices):
    """Read the chosen frames of one video from its JPG folder.

    Args:
        video_dir (Path): Folder holding 000000.jpg, 000001.jpg, ...
        indices (list of int): Frame numbers to read (length T).

    Returns:
        Tensor: (T, 3, H, W), uint8, RGB.
    """
    frames = [read_image(str(video_dir / FRAME_NAME_FORMAT.format(i)), ImageReadMode.RGB)
              for i in indices]
    return torch.stack(frames)   # (T, 3, H, W)


def transform_clip(clip, size, train):
    """Crop to a square, optionally flip, and normalize a clip; same crop/flip for every frame.

    Train: random crop + random horizontal flip. Eval: center crop.
    Every frame gets the same crop and flip, so motion between frames is kept.

    Args:
        clip (Tensor): (T, 3, H, W), uint8, with H >= size and W >= size.
        size (int): Output height and width.
        train (bool): Use the random (training) augmentation.

    Returns:
        Tensor: (T, 3, size, size), float32, ImageNet-normalized.

    Raises:
        AssertionError: If the frames are smaller than `size`.
    """
    _, _, h, w = clip.shape
    assert h >= size and w >= size, f"frames are {h}x{w}, smaller than the crop size {size}"

    # crop
    if train:
        top = np.random.randint(h - size + 1)
        left = np.random.randint(w - size + 1)
    else:
        top, left = (h - size) // 2, (w - size) // 2
    clip = clip[:, :, top:top + size, left:left + size]   # (T, 3, size, size)

    # flip
    if train and np.random.rand() < FLIP_PROB:
        clip = clip.flip(dims=[3])

    # normalize
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)   # (1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)     # (1, 3, 1, 1)
    return (clip.float() / 255 - mean) / std


class VideoDataset(Dataset):
    """One item = one video, as a clip of T frames plus its class label.

    Args:
        csv_path (str or Path): Split file (train.csv / val.csv / test.csv).
        frames_dir (str or Path): Root of the extracted frames (config `frames_dir`).
        n_segment (int): T, frames per clip.
        image_size (int): Square crop size fed to the model.
        train (bool): Random sampling + augmentation (True) or deterministic (False).

    Item:
        tuple: clip (Tensor, (T, 3, image_size, image_size), float, normalized),
            label (int).
    """

    def __init__(self, csv_path, frames_dir, n_segment, image_size, train):
        self.videos = read_split_csv(csv_path)
        self.frames_dir = Path(frames_dir)
        self.n_segment = n_segment
        self.image_size = image_size
        self.train = train

    def __len__(self):
        """Number of videos in the split.

        Returns:
            int: Number of videos.
        """
        return len(self.videos)

    def __getitem__(self, index):
        """Load and transform one video's clip.

        Args:
            index (int): Video index in [0, len(self)).

        Returns:
            tuple: clip (Tensor, (T, 3, image_size, image_size), float), label (int).
        """
        video_id, label, num_frames = self.videos[index]
        indices = sample_indices(num_frames, self.n_segment, self.train)
        clip = load_frames(self.frames_dir / video_id, indices)       # (T, 3, H, W), uint8
        return transform_clip(clip, self.image_size, self.train), label


# ---------------------------------------------------------------------------
# Self-test:  python dataset.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(0)

    # 1) eval sampling picks the segment centers: 32 frames / 8 segments -> 2, 6, ..., 30
    expected = [2, 6, 10, 14, 18, 22, 26, 30]
    got = sample_indices(32, 8, train=False)
    assert got == expected, f"expected {expected}, got {got}"
    print("[1] sample_indices eval ... OK")

    # 2) train sampling: one index inside each 4-frame segment; short videos repeat frames
    for _ in range(100):
        got = sample_indices(32, 8, train=True)
        assert all(4 * i <= idx < 4 * (i + 1) for i, idx in enumerate(got)), f"bad indices {got}"
    got = sample_indices(3, 8, train=True)
    assert len(got) == 8 and min(got) >= 0 and max(got) <= 2, f"short video indices {got}"
    print("[2] sample_indices train + short video ... OK")

    # 3) transform_clip: shapes, deterministic center crop, same crop/flip on every frame
    frame = torch.randint(0, 256, (1, 3, 72, 80), dtype=torch.uint8)
    clip = frame.repeat(4, 1, 1, 1)                         # (T=4, 3, 72, 80), identical frames
    out = transform_clip(clip, 64, train=True)
    assert out.shape == (4, 3, 64, 64) and out.dtype == torch.float32, f"bad output {out.shape}"
    assert all(torch.equal(out[0], out[t]) for t in range(4)), "frames got different crop/flip"
    same = torch.equal(transform_clip(clip, 64, False), transform_clip(clip, 64, False))
    assert same, "eval transform is not deterministic"
    print("[3] transform_clip shapes + consistency ... OK")

    # 4) normalization: a frame equal to the ImageNet mean becomes ~0
    mean_frame = (torch.tensor(IMAGENET_MEAN) * 255).round().to(torch.uint8)
    mean_clip = mean_frame.view(1, 3, 1, 1).expand(2, 3, 64, 64)    # (2, 3, 64, 64)
    normalized = transform_clip(mean_clip, 64, train=False)
    assert normalized.abs().max() < 0.01, f"mean frame should map to ~0, max {normalized.abs().max()}"
    print("[4] transform_clip normalization ... OK")

    # 5) a crop larger than the frame must fail loudly
    try:
        transform_clip(clip, 100, train=False)
        raise RuntimeError("expected an AssertionError for crop 100 > frame 72x80")
    except AssertionError:
        pass
    print("[5] too-small frames rejected ... OK")

    with tempfile.TemporaryDirectory() as tmp:
        frames_dir, splits_dir = make_dummy_dataset(tmp, videos_per_class=2, num_frames=8)

        # 6) read_split_csv + load_frames on the fake dataset
        rows = read_split_csv(splits_dir / "train.csv")
        assert len(rows) == 4 and {label for _, label, _ in rows} == {0, 1}, f"bad rows {rows}"
        frames = load_frames(frames_dir / rows[0][0], [0, 3, 7])
        assert (frames.shape == (3, 3, 72, 80) and frames.dtype == torch.uint8), (
            f"bad frames {frames.shape} {frames.dtype}")
        print("[6] read_split_csv + load_frames ... OK")

        # 7) VideoDataset items and a DataLoader batch have the model's input shape
        ds = VideoDataset(splits_dir / "val.csv", frames_dir, n_segment=4, image_size=64,
                          train=False)
        clip, label = ds[0]
        assert (len(ds) == 4 and clip.shape == (4, 3, 64, 64) and label == 0), (
            f"bad item: len {len(ds)}, clip {tuple(clip.shape)}, label {label}")
        clips, labels = next(iter(torch.utils.data.DataLoader(ds, batch_size=4)))
        assert (clips.shape == (4, 4, 3, 64, 64) and labels.tolist() == [0, 0, 1, 1]), (
            f"bad batch: {tuple(clips.shape)}, labels {labels.tolist()}")
        print("[7] VideoDataset + DataLoader batch ... OK")

    print("All 7 checks passed.")
