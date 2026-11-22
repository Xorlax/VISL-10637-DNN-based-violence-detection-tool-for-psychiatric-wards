"""
One-time data preparation: extract frames from the raw videos and write the split CSVs.

Two raw_dir layouts are supported (class folder names must match config `classes`):

  A) split folders (e.g. RWF-2000):        B) class folders only (e.g. RLVS):
       raw_dir/train/<class>/*.avi              raw_dir/<class>/*.mp4
       raw_dir/val/<class>/*.avi
       [raw_dir/test/<class>/*.avi]

  Layout A: if there is no test folder, the official val folder becomes the test split
  (so results are comparable with papers that report on it), and a new val split is
  taken from train (val_ratio).
  Layout B: test (test_ratio) and then val (val_ratio of the rest) are taken at random.
  All random splits are stratified by class and use config `seed`.

Output:
  frames_dir/<split>/<class>/<video name>/000000.jpg ...   (frames_per_video evenly spaced
                                                           frames, short side frame_short_side)
  splits_dir/train.csv, val.csv, test.csv                  (video_id, label, num_frames)

Usage:
  python prepare_data.py [--config config.yaml]
  python prepare_data.py --self-test
"""

import argparse
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np

from utils import FRAME_NAME_FORMAT, load_config, write_dummy_video, write_split_csv


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mpg", ".mpeg", ".mov", ".mkv", ".webm"}
SPLIT_NAMES = ("train", "val", "test")
PRINT_EVERY = 50   # progress line every N videos


def list_class_videos(folder, classes):
    """List the videos of every class folder inside `folder`.

    Args:
        folder (Path): Folder holding one sub-folder per class.
        classes (list of str): Class folder names; list index = label.

    Returns:
        list of tuple: (video_path (Path), label (int)), sorted by path.

    Raises:
        FileNotFoundError: If a class folder is missing.
        ValueError: If `folder` holds folders that are not classes.
    """
    unexpected = sorted(p.name for p in folder.iterdir() if p.is_dir() and p.name not in classes)
    if unexpected:
        raise ValueError(f"{folder}: folders {unexpected} are not in classes {classes}")
    videos = []
    for label, name in enumerate(classes):
        class_dir = folder / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"class folder missing: {class_dir}")
        files = sorted(p for p in class_dir.iterdir() if p.is_file())
        found = [p for p in files if p.suffix.lower() in VIDEO_EXTENSIONS]
        if len(found) < len(files):
            print(f"note: ignored {len(files) - len(found)} non-video files in {class_dir}")
        videos += [(p, label) for p in found]
    return videos


def carve_split(videos, ratio, rng):
    """Move a random `ratio` of the videos of each class into a new split.

    Args:
        videos (list of tuple): (video_path, label) pairs.
        ratio (float): Fraction in (0, 1) to move, per class.
        rng (random.Random): Random generator (seeded by the caller).

    Returns:
        tuple:
            rest (list of tuple): Videos that stay.
            carved (list of tuple): Videos moved to the new split.

    Raises:
        ValueError: If `ratio` is not in (0, 1) or the new split would be empty.
    """
    if not 0 < ratio < 1:
        raise ValueError(f"split ratio must be in (0, 1), got {ratio}")
    rest, carved = [], []
    for label in sorted({label for _, label in videos}):
        class_videos = [v for v in videos if v[1] == label]
        rng.shuffle(class_videos)
        n_carved = round(len(class_videos) * ratio)
        carved += class_videos[:n_carved]
        rest += class_videos[n_carved:]
    if not carved:
        raise ValueError(f"ratio {ratio} of {len(videos)} videos gives an empty split")
    return sorted(rest), sorted(carved)


def make_splits(raw_dir, classes, val_ratio, test_ratio, seed):
    """Assign every raw video to train / val / test (see the module docstring for the rules).

    Args:
        raw_dir (Path): Raw video root (layout A or B).
        classes (list of str): Class folder names.
        val_ratio (float): Fraction of the training videos moved to val.
        test_ratio (float): Fraction of all videos moved to test (layout B only).
        seed (int): Seed for the random splits.

    Returns:
        dict: split name -> list of (video_path (Path), label (int)).
    """
    rng = random.Random(seed)
    folder_splits = {s: raw_dir / s for s in SPLIT_NAMES if (raw_dir / s).is_dir()}

    if folder_splits:   # layout A
        if "train" not in folder_splits:
            raise FileNotFoundError(f"{raw_dir} has split folders but no 'train' folder")
        splits = {s: list_class_videos(folder, classes) for s, folder in folder_splits.items()}
        if "test" not in splits:
            if "val" in splits:
                print("note: no test folder, using the official val folder as the test split")
                splits["test"] = splits.pop("val")
            else:
                splits["train"], splits["test"] = carve_split(splits["train"], test_ratio, rng)
    else:               # layout B
        all_videos = list_class_videos(raw_dir, classes)
        train, test = carve_split(all_videos, test_ratio, rng)
        splits = {"train": train, "test": test}

    if "val" not in splits:
        splits["train"], splits["val"] = carve_split(splits["train"], val_ratio, rng)
    return splits


def resize_short_side(frame, short_side):
    """Resize a frame so its shorter side is `short_side` pixels, keeping the aspect ratio.

    Also used by infer.py, so live frames get exactly the training preprocessing.

    Args:
        frame (np.ndarray): (H, W, 3), uint8, BGR (as read by OpenCV).
        short_side (int): Target length of the shorter side.

    Returns:
        np.ndarray: (H', W', 3), uint8, BGR, with min(H', W') == short_side.
    """
    h, w = frame.shape[:2]
    scale = short_side / min(h, w)
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def extract_frames(video_path, out_dir, frames_per_video, short_side):
    """Decode a video, keep evenly spaced frames, resize them, and save them as JPGs.

    All frames are decoded because the frame count stored in video headers is often wrong.

    Args:
        video_path (Path): Input video.
        out_dir (Path): Output folder (created), gets 000000.jpg, 000001.jpg, ...
        frames_per_video (int): Frames to keep; shorter videos keep all their frames.
        short_side (int): Frames are resized so min(H, W) == short_side.

    Returns:
        int: Number of frames saved.

    Raises:
        RuntimeError: If the video can't be opened or has no frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()                      # (H, W, 3), uint8, BGR
        if not ok:
            break
        frames.append(resize_short_side(frame, short_side))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames could be decoded from: {video_path}")

    # keep the center frame of frames_per_video equal-length pieces of the video
    n = len(frames)
    keep = range(n) if n <= frames_per_video else [
        int((i + 0.5) * n / frames_per_video) for i in range(frames_per_video)]

    out_dir.mkdir(parents=True)
    for out_index, frame_index in enumerate(keep):
        cv2.imwrite(str(out_dir / FRAME_NAME_FORMAT.format(out_index)), frames[frame_index])
    return len(keep)


def prepare(cfg):
    """Run the whole preparation: split the videos, extract frames, write the CSVs.

    Args:
        cfg (dict): Config with raw_dir, frames_dir, splits_dir, classes, frames_per_video,
            frame_short_side, val_ratio, test_ratio, seed.

    Returns:
        dict: split name -> list of (video_id (str), label (int), num_frames (int)),
            the rows written to the CSVs.

    Raises:
        FileExistsError: If frames_dir already has content (delete it to re-run).
        ValueError: If two videos of one class map to the same name.
    """
    raw_dir, frames_dir = Path(cfg["raw_dir"]), Path(cfg["frames_dir"])
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")
    if frames_dir.exists() and any(frames_dir.iterdir()):
        raise FileExistsError(f"{frames_dir} is not empty; delete it to prepare the data again")

    splits = make_splits(raw_dir, cfg["classes"], cfg["val_ratio"], cfg["test_ratio"], cfg["seed"])
    total = sum(len(videos) for videos in splits.values())
    print(f"videos: {total} (" + ", ".join(f"{s} {len(splits[s])}" for s in SPLIT_NAMES) + ")")

    all_rows = {}
    done = 0
    for split in SPLIT_NAMES:
        rows = []
        for video_path, label in splits[split]:
            video_id = f"{split}/{cfg['classes'][label]}/{video_path.stem}"
            if any(row[0] == video_id for row in rows):
                raise ValueError(f"two videos map to {video_id} (same name, other extension?)")
            num_frames = extract_frames(video_path, frames_dir / video_id,
                                        cfg["frames_per_video"], cfg["frame_short_side"])
            rows.append((video_id, label, num_frames))
            done += 1
            if done % PRINT_EVERY == 0 or done == total:
                print(f"  extracted {done}/{total}")
        write_split_csv(Path(cfg["splits_dir"]) / f"{split}.csv", rows)
        all_rows[split] = rows
    print(f"done: frames in {frames_dir}, splits in {cfg['splits_dir']}")
    return all_rows


def _self_test():
    """Check splitting and frame extraction on tiny generated videos.

    Returns:
        None
    """
    classes = ["NonFight", "Fight"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # layout B: 5 videos per class; class 0 videos have 12 frames, class 1 have 5
        for label, name in enumerate(classes):
            for k in range(5):
                write_dummy_video(tmp / "rawB" / name / f"v{k}.avi", 12 if label == 0 else 5,
                                  height=40, width=60)
        (tmp / "rawB" / "Fight" / "notes.txt").write_text("not a video")
        cfg = {"raw_dir": tmp / "rawB", "frames_dir": tmp / "framesB",
               "splits_dir": tmp / "splitsB", "classes": classes, "frames_per_video": 8,
               "frame_short_side": 32, "val_ratio": 0.25, "test_ratio": 0.2, "seed": 0}

        # 1) list_class_videos finds the 10 videos and skips the .txt file
        videos = list_class_videos(tmp / "rawB", classes)
        assert len(videos) == 10, f"expected 10 videos, got {len(videos)}"
        print("[1] list_class_videos ... OK")

        # 2) carve_split is stratified: 20% of 5 per class = 1 per class
        rest, carved = carve_split(videos, 0.2, random.Random(0))
        assert sorted(label for _, label in carved) == [0, 1], f"not stratified: {carved}"
        assert len(rest) == 8, f"expected 8 left, got {len(rest)}"
        print("[2] carve_split ... OK")

        # 3) prepare on layout B: split sizes per class are test 1, val 1, train 3
        rows = prepare(cfg)
        for split, expected in (("train", 6), ("val", 2), ("test", 2)):
            assert len(rows[split]) == expected, f"{split}: expected {expected}, got {len(rows[split])}"
        all_ids = [r[0] for split_rows in rows.values() for r in split_rows]
        assert len(set(all_ids)) == 10, "a video appears in two splits"
        print("[3] prepare layout B split sizes ... OK")

        # 4) extract_frames: long videos keep frames_per_video, short keep all, short side 32
        for split_rows in rows.values():
            for video_id, label, num_frames in split_rows:
                expected = 8 if label == 0 else 5
                assert num_frames == expected, f"{video_id}: expected {expected}, got {num_frames}"
                jpgs = sorted((tmp / "framesB" / video_id).glob("*.jpg"))
                assert len(jpgs) == expected, f"{video_id}: {len(jpgs)} jpgs on disk"
                h, w = cv2.imread(str(jpgs[0])).shape[:2]
                assert (h, w) == (32, 48), f"{video_id}: frame size {(h, w)}, expected (32, 48)"
        for shape, expected in (((40, 60, 3), (32, 48, 3)), ((90, 30, 3), (96, 32, 3))):
            resized = resize_short_side(np.zeros(shape, dtype=np.uint8), 32)
            assert resized.shape == expected, f"{shape} -> {resized.shape}, expected {expected}"
        print("[4] extract_frames count + size, resize_short_side ... OK")

        # 5) layout A without a test folder: the official val folder becomes test
        for split, n in (("train", 4), ("val", 2)):
            for name in classes:
                for k in range(n):
                    write_dummy_video(tmp / "rawA" / split / name / f"v{k}.avi", 6, 40, 60)
        splits = make_splits(tmp / "rawA", classes, val_ratio=0.25, test_ratio=0.2, seed=0)
        assert all("rawA" in str(p) and "val" in p.parts for p, _ in splits["test"]), (
            "test split should be the official val folder")
        sizes = {s: len(v) for s, v in splits.items()}
        assert sizes == {"train": 6, "val": 2, "test": 4}, f"unexpected sizes {sizes}"
        print("[5] make_splits layout A ... OK")

        # 6) bad inputs fail loudly: re-running onto full frames_dir, unknown class folder
        try:
            prepare(cfg)
            raise RuntimeError("expected FileExistsError for a non-empty frames_dir")
        except FileExistsError:
            pass
        (tmp / "rawB" / "Other").mkdir()
        try:
            list_class_videos(tmp / "rawB", classes)
            raise RuntimeError("expected ValueError for the unknown 'Other' folder")
        except ValueError:
            pass
        print("[6] bad inputs rejected ... OK")

    print("All 6 checks passed.")


def main():
    """Parse the command line and run the data preparation (or the self-test).

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Extract frames and write split CSVs")
    parser.add_argument("--config", default="config.yaml", help="path to the YAML config")
    parser.add_argument("--self-test", action="store_true", help="run the self-test and exit")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        prepare(load_config(args.config))


if __name__ == "__main__":
    main()
