"""
Measure the model's speed on this machine: latency of one T-frame clip (batch 1).

The weights don't change the speed, so the model is built from config.yaml with random
weights (no checkpoint or download needed).

Usage:
    python benchmark.py [--config config.yaml] [--warmup 5] [--runs 30]
    python benchmark.py --self-test
"""

import argparse
import statistics
import time

import torch

from utils import build_model, get_device, load_config, set_seed


def benchmark(model, n_segment, image_size, device, warmup, runs):
    """Time the forward pass of one clip.

    Args:
        model (nn.Module): Video classifier, (1, T, 3, H, W) -> logits (1, K), on `device`.
        n_segment (int): T, frames per clip.
        image_size (int): Frame height and width.
        device (torch.device): Device to run on.
        warmup (int): Untimed runs first (lets cuDNN / caches settle), >= 0.
        runs (int): Timed runs, >= 1.

    Returns:
        dict: mean_ms, median_ms, min_ms (latency per clip, milliseconds),
            clips_per_s (1000 / mean_ms) and frames_per_s (clips_per_s * T).

    Raises:
        ValueError: If runs < 1 or warmup < 0.
    """
    if runs < 1 or warmup < 0:
        raise ValueError(f"need runs >= 1 and warmup >= 0, got runs={runs}, warmup={warmup}")
    model.eval()
    clip = torch.randn(1, n_segment, 3, image_size, image_size, device=device)   # (1, T, 3, H, W)
    times_ms = []
    with torch.no_grad():
        for i in range(warmup + runs):
            if device.type == "cuda":
                torch.cuda.synchronize()   # GPU calls are async: wait before reading the clock
            start = time.perf_counter()
            model(clip)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= warmup:
                times_ms.append((time.perf_counter() - start) * 1000)
    mean_ms = statistics.mean(times_ms)
    return {"mean_ms": mean_ms, "median_ms": statistics.median(times_ms),
            "min_ms": min(times_ms), "clips_per_s": 1000 / mean_ms,
            "frames_per_s": 1000 / mean_ms * n_segment}


def _self_test():
    """Check benchmark on a tiny model (CPU).

    Returns:
        None
    """
    set_seed(0)
    cfg = {"classes": ["NonFight", "Fight"], "arch": "small", "n_segment": 4, "shift_div": 8,
           "causal_shift": False, "dropout": 0.5, "pretrained": False}
    model = build_model(cfg)

    # 1) the numbers are positive and consistent with each other
    r = benchmark(model, n_segment=4, image_size=64, device=torch.device("cpu"), warmup=1, runs=5)
    assert 0 < r["min_ms"] <= r["median_ms"], f"min/median inconsistent: {r}"
    assert abs(r["clips_per_s"] - 1000 / r["mean_ms"]) < 1e-9, f"clips/s wrong: {r}"
    assert abs(r["frames_per_s"] - 4 * r["clips_per_s"]) < 1e-9, f"frames/s wrong: {r}"
    print("[1] benchmark numbers ... OK")

    # 2) invalid run counts fail loudly
    try:
        benchmark(model, 4, 64, torch.device("cpu"), warmup=0, runs=0)
        raise RuntimeError("expected ValueError for runs=0")
    except ValueError:
        pass
    print("[2] invalid runs rejected ... OK")

    print("All 2 checks passed.")


def main():
    """Parse the command line and run the benchmark (or the self-test).

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Measure TSM-MobileNetV3 latency")
    parser.add_argument("--config", default="config.yaml", help="path to the YAML config")
    parser.add_argument("--warmup", type=int, default=5, help="untimed runs before timing")
    parser.add_argument("--runs", type=int, default=30, help="timed runs")
    parser.add_argument("--self-test", action="store_true", help="run the self-test and exit")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return

    cfg = load_config(args.config)
    device = get_device()
    model = build_model(dict(cfg, pretrained=False)).to(device)   # weights don't affect speed
    r = benchmark(model, cfg["n_segment"], cfg["image_size"], device, args.warmup, args.runs)
    print(f"{cfg['arch']} MobileNetV3 + TSM, T={cfg['n_segment']}, {cfg['image_size']}px, "
          f"device {device}, {args.runs} runs")
    print(f"  latency per clip: mean {r['mean_ms']:.1f} ms, median {r['median_ms']:.1f} ms, "
          f"min {r['min_ms']:.1f} ms")
    print(f"  throughput: {r['clips_per_s']:.1f} clips/s = {r['frames_per_s']:.1f} frames/s")


if __name__ == "__main__":
    main()
