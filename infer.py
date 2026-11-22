"""
Real-time violence detection on a video file or a webcam, with a trained checkpoint.

How it works: a frame is taken every `step` frames (so T frames span infer_clip_seconds,
like a training clip spans a whole video), preprocessed exactly as in training, and kept
in a sliding window of the last T frames. Each time a new frame enters a full window, the
model classifies the window and the label is drawn on the video.

The whole window is re-run each time, so the bidirectional shift works as in training.
(TSM's online mode, which caches shifted features and needs causal_shift=True, is not used.)

Usage:
    python infer.py --source path/to/video.mp4   [--config config.yaml] [--checkpoint ...]
    python infer.py --source 0                   (webcam 0; press q to quit)
    python infer.py --self-test
"""

import argparse
import tempfile
import time
from collections import deque
from pathlib import Path

import cv2  # type: ignore[import-not-found,import-untyped]
import torch  # type: ignore[import-not-found,import-untyped]

from dataset import transform_clip
from prepare_data import resize_short_side
from utils import build_model, get_device, load_checkpoint, load_config, set_seed, write_dummy_video


WINDOW_NAME = "violence detection (q = quit)"
TEXT_ORIGIN = (10, 30)           # top-left corner of the label text, pixels
TEXT_SCALE = 0.8
TEXT_THICKNESS = 2
POSITIVE_COLOR = (0, 0, 255)     # BGR red for the positive class (e.g. Fight)
NEGATIVE_COLOR = (0, 255, 0)     # BGR green otherwise


def frame_step(fps, clip_seconds, n_segment):
    """Number of video frames between two frames of the clip.

    Args:
        fps (float): Video frame rate (> 0).
        clip_seconds (float): Time one T-frame clip should cover.
        n_segment (int): T, frames per clip.

    Returns:
        int: Step >= 1, so that T frames span about clip_seconds.
    """
    return max(1, round(fps * clip_seconds / n_segment))


def preprocess_frame(frame, short_side, size):
    """Turn one OpenCV frame into a model-ready frame, exactly as in training.

    Training frames are resized by prepare_data.py and then center-cropped and normalized
    by dataset.transform_clip (eval mode); the same two functions are used here.

    Args:
        frame (np.ndarray): (H, W, 3), uint8, BGR.
        short_side (int): Resize target of the shorter side (config frame_short_side).
        size (int): Square crop size (config image_size).

    Returns:
        Tensor: (3, size, size), float, RGB, ImageNet-normalized.
    """
    rgb = cv2.cvtColor(resize_short_side(frame, short_side), cv2.COLOR_BGR2RGB)   # (H', W', 3)
    clip = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)    # (1, 3, H', W'), uint8
    return transform_clip(clip, size, train=False)[0]              # (3, size, size)


@torch.no_grad()
def classify_clip(model, frames, device):
    """Class probabilities for one clip.

    Args:
        model (nn.Module): Video classifier in eval mode, (B, T, 3, H, W) -> logits (B, K).
        frames (list of Tensor): T preprocessed frames, each (3, H, W), float.
        device (torch.device): Device the model is on.

    Returns:
        Tensor: (K,), float, softmax probabilities (on CPU).
    """
    clip = torch.stack(list(frames)).unsqueeze(0).to(device)   # (1, T, 3, H, W)
    return model(clip).softmax(dim=1)[0].cpu()                  # (K,)


def run(model, train_cfg, source, clip_seconds, device, fps=None, display=True):
    """Read a video / webcam, classify a sliding window of frames, optionally show it.

    Args:
        model (nn.Module): Trained model in eval mode, on `device`.
        train_cfg (dict): Config stored in the checkpoint (classes, n_segment,
            frame_short_side, image_size).
        source (str or int): Video file path, or webcam index.
        clip_seconds (float): Time one clip should cover (config infer_clip_seconds).
        device (torch.device): Device the model is on.
        fps (float, optional): Frame rate to assume; None = read it from the video.
        display (bool): Show the video with the label drawn on it. Default True.

    Returns:
        list of tuple: (frame_index (int), probs (Tensor, (K,))) for every prediction.

    Raises:
        RuntimeError: If the source can't be opened.
        ValueError: If the video reports no frame rate and none is given.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video source: {source}")
    fps = fps or cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise ValueError(f"{source} reports no frame rate; pass it with --fps")
    n_segment, classes = train_cfg["n_segment"], train_cfg["classes"]
    step = frame_step(fps, clip_seconds, n_segment)
    print(f"source fps {fps:.1f}: one clip frame every {step} video frames")

    window = deque(maxlen=n_segment)
    predictions = []
    text, color = "collecting frames...", NEGATIVE_COLOR
    frame_index = 0
    start = time.perf_counter()
    while True:
        ok, frame = cap.read()                  # (H, W, 3), uint8, BGR
        if not ok:
            break

        # add every step-th frame to the window; classify once the window is full
        if frame_index % step == 0:
            window.append(preprocess_frame(frame, train_cfg["frame_short_side"],
                                           train_cfg["image_size"]))
            if len(window) == n_segment:
                probs = classify_clip(model, window, device)
                predictions.append((frame_index, probs))
                pred = probs.argmax().item()
                text = f"{classes[pred]} {probs[pred]:.2f}"
                color = POSITIVE_COLOR if pred == 1 else NEGATIVE_COLOR

        if display:
            speed = (frame_index + 1) / (time.perf_counter() - start)
            cv2.putText(frame, f"{text}   {speed:.0f} fps", TEXT_ORIGIN,
                        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, color, TEXT_THICKNESS)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_index += 1

    cap.release()
    if display:
        cv2.destroyAllWindows()
    return predictions


def _self_test():
    """Check preprocessing, frame stepping and the sliding-window loop (no display).

    Returns:
        None
    """
    set_seed(0)
    device = torch.device("cpu")

    # 1) frame_step: 30 fps, 5 s, 8 frames -> 18.75 -> 19; never below 1
    assert frame_step(30, 5.0, 8) == 19, f"got {frame_step(30, 5.0, 8)}"
    assert frame_step(25, 1.0, 4) == 6, f"got {frame_step(25, 1.0, 4)}"
    assert frame_step(10, 0.1, 8) == 1, f"got {frame_step(10, 0.1, 8)}"
    print("[1] frame_step ... OK")

    # 2) preprocess_frame: shape, and BGR -> RGB (a pure red BGR frame -> max in channel 0)
    red = torch.zeros(40, 60, 3, dtype=torch.uint8)
    red[..., 2] = 255                                             # BGR red
    out = preprocess_frame(red.numpy(), short_side=36, size=32)
    assert out.shape == (3, 32, 32), f"bad shape {tuple(out.shape)}"
    expected_r = (1 - 0.485) / 0.229                              # normalized value of 255 in R
    assert torch.allclose(out[0], torch.full((32, 32), expected_r), atol=1e-4), (
        "channel 0 should be the red channel at full intensity")
    print("[2] preprocess_frame ... OK")

    train_cfg = {"classes": ["NonFight", "Fight"], "arch": "small", "n_segment": 4,
                 "shift_div": 8, "causal_shift": False, "dropout": 0.5, "pretrained": False,
                 "image_size": 32, "frame_short_side": 36}
    model = build_model(train_cfg).eval()

    # 3) classify_clip: a probability vector over the 2 classes
    probs = classify_clip(model, [out] * 4, device)
    assert probs.shape == (2,) and abs(probs.sum().item() - 1) < 1e-5, f"bad probs {probs}"
    print("[3] classify_clip ... OK")

    with tempfile.TemporaryDirectory() as tmp:
        # 4) run: 40 frames at 25 fps, 1 s clips of 4 frames -> step 6; window frames
        #    0, 6, 12, 18 first fill the window, so predictions come at 18, 24, 30, 36
        video = Path(tmp) / "v.avi"
        write_dummy_video(video, num_frames=40, height=40, width=60, fps=25)
        preds = run(model, train_cfg, str(video), clip_seconds=1.0, device=device,
                    display=False)
        frames = [i for i, _ in preds]
        assert frames == [18, 24, 30, 36], f"predictions at frames {frames}"
        print("[4] run sliding window ... OK")

        # 5) a missing source fails loudly
        try:
            run(model, train_cfg, str(Path(tmp) / "missing.avi"), 1.0, device, display=False)
            raise RuntimeError("expected RuntimeError for a missing video")
        except RuntimeError as err:
            assert "cannot open" in str(err), f"unexpected error: {err}"
        print("[5] missing source rejected ... OK")

    print("All 5 checks passed.")


def main():
    """Parse the command line and run the demo (or the self-test).

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Real-time violence detection demo")
    parser.add_argument("--source", help="video file path, or webcam index such as 0")
    parser.add_argument("--config", default="config.yaml", help="path to the YAML config")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint to use (default: <checkpoint_dir>/best.pt)")
    parser.add_argument("--fps", type=float, default=None,
                        help="frame rate to assume, if the source does not report one")
    parser.add_argument("--self-test", action="store_true", help="run the self-test and exit")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.source is None:
        parser.error("--source is required (a video path or a webcam index)")

    cfg = load_config(args.config)
    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint or Path(cfg["checkpoint_dir"]) / "best.pt",
                                  device)
    source = int(args.source) if args.source.isdigit() else args.source
    run(model, ckpt["config"], source, cfg["infer_clip_seconds"], device, fps=args.fps)


if __name__ == "__main__":
    main()
