"""
Hyperparameter sweep: run train.py's train() once per combination of config values.

Every run gets its own folder runs_dir/<name>/ with its config.yaml, last.pt, best.pt and
history.json. After every run, runs_dir/summary.csv is rewritten with one row per run
(its values, best epoch, val accuracy / F1 at that epoch), sorted by val accuracy.
Runs whose history.json already exists are skipped, so an interrupted sweep can be resumed
by running the same command again.

Pick the best run by VALIDATION score, then evaluate only that one with test.py:
    python test.py --checkpoint runs/<name>/best.pt

Usage:
    python sweep.py                                   (default grid, see --grid)
    python sweep.py --grid lr=0.003,0.01,0.03 weight_decay=0.0001,0.0005 seed=0,1,2
    python sweep.py --self-test
"""

import argparse
import csv
import itertools
import json
import tempfile
from pathlib import Path

import yaml  # type: ignore[import-not-found,import-untyped]

from train import train
from utils import load_config, make_dummy_dataset


DEFAULT_GRID = ["lr=0.003,0.01,0.03", "weight_decay=0.0001,0.0005"]


def parse_grid(items, base_cfg):
    """Turn "key=v1,v2,..." strings into lists of values with the config's types.

    Args:
        items (list of str): E.g. ["lr=0.003,0.01", "arch=small,large"].
        base_cfg (dict): Base config; every key must exist in it, and the values are
            converted to the type of the base value (int / float / bool / str).

    Returns:
        dict: key -> list of values, in the order given.

    Raises:
        ValueError: If an item is malformed or a value doesn't fit the key's type.
        KeyError: If a key is not in the config (catches typos).
    """
    grid = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"grid item must look like key=v1,v2, got {item!r}")
        key, values = item.split("=", 1)
        if key not in base_cfg:
            raise KeyError(f"{key!r} is not a config key")
        base_type = type(base_cfg[key])
        parsed = []
        for text in values.split(","):
            value = yaml.safe_load(text)   # "0.01" -> 0.01, "true" -> True, "small" -> "small"
            # YAML reads "1e-4" (no dot) as a string, and "1" as an int: convert for floats
            if base_type is float and isinstance(value, (int, str)) and not isinstance(value, bool):
                value = float(value)
            if type(value) is not base_type:
                raise ValueError(f"{key}={text!r}: expected {base_type.__name__} like the config "
                                 f"value {base_cfg[key]!r}, got {type(value).__name__}")
            parsed.append(value)
        grid[key] = parsed
    return grid


def run_name(values):
    """Folder name of one run, e.g. "lr0.01_weight_decay0.0005".

    Args:
        values (dict): The swept key -> value of this run.

    Returns:
        str: Name built from the keys and values.
    """
    return "_".join(f"{key}{value}" for key, value in values.items())


def sweep(base_cfg, grid, runs_dir):
    """Train once per combination of grid values and write the summary table.

    Args:
        base_cfg (dict): Config every run starts from.
        grid (dict): key -> list of values (output of parse_grid).
        runs_dir (str or Path): Folder for the run folders and summary.csv.

    Returns:
        list of dict: Summary rows (name, swept values, best_epoch, val_accuracy, val_f1,
            val_loss), sorted by val_accuracy, best first.
    """
    runs_dir = Path(runs_dir)
    combos = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    print(f"sweep: {len(combos)} runs over {list(grid)}")
    rows = []
    for i, values in enumerate(combos, 1):
        name = run_name(values)
        run_dir = runs_dir / name
        history_path = run_dir / "history.json"

        # train (or reuse a finished run)
        if history_path.is_file():
            print(f"\n[{i}/{len(combos)}] {name}: already done, skipping")
            history = json.loads(history_path.read_text(encoding="utf-8"))
        else:
            print(f"\n[{i}/{len(combos)}] {name}")
            cfg = dict(base_cfg, **values, checkpoint_dir=str(run_dir))
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False),
                                                 encoding="utf-8")
            history = train(cfg)
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        # best epoch = the one best.pt was saved at (first epoch with the max val accuracy)
        best = max(history, key=lambda h: h["val_accuracy"])
        rows.append({"name": name, **values, "best_epoch": best["epoch"],
                     "val_accuracy": best["val_accuracy"], "val_f1": best["val_f1"],
                     "val_loss": best["val_loss"]})
        rows.sort(key=lambda r: r["val_accuracy"], reverse=True)
        _write_summary(runs_dir / "summary.csv", rows)

    print(f"\nsummary ({runs_dir / 'summary.csv'}), best first:")
    for r in rows:
        print(f"  {r['name']:<40} val_acc={r['val_accuracy']:.4f}  val_f1={r['val_f1']:.4f}  "
              f"(epoch {r['best_epoch']})")
    return rows


def _write_summary(path, rows):
    """Write the summary rows as a CSV table.

    Args:
        path (Path): Output CSV path.
        rows (list of dict): Summary rows, all with the same keys.

    Returns:
        None
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _self_test():
    """Check grid parsing, a 2-run sweep, the summary and resuming, on a tiny fake dataset.

    Returns:
        None
    """
    base_cfg = {"classes": ["NonFight", "Fight"], "arch": "small", "n_segment": 4,
                "shift_div": 8, "causal_shift": False, "dropout": 0.5, "pretrained": False,
                "image_size": 64, "seed": 0, "batch_size": 2, "num_workers": 0, "epochs": 1,
                "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005, "amp": False}

    # 1) parse_grid: types follow the config, "1e-4" and "1" become floats
    grid = parse_grid(["weight_decay=1e-4,0.0005", "lr=1", "arch=small", "amp=true"], base_cfg)
    assert grid == {"weight_decay": [0.0001, 0.0005], "lr": [1.0], "arch": ["small"],
                    "amp": [True]}, f"bad grid {grid}"
    assert all(type(v) is float for v in grid["weight_decay"] + grid["lr"]), "floats expected"
    print("[1] parse_grid types ... OK")

    # 2) parse_grid fails loudly on a typo, a wrong type and a malformed item
    for items, error in ((["lerning_rate=0.1"], KeyError), (["epochs=0.5"], ValueError),
                         (["lr"], ValueError)):
        try:
            parse_grid(items, base_cfg)
            raise RuntimeError(f"expected {error.__name__} for {items}")
        except error:
            pass
    print("[2] parse_grid bad items rejected ... OK")

    # 3) run_name
    assert run_name({"lr": 0.01, "seed": 1}) == "lr0.01_seed1", run_name({"lr": 0.01, "seed": 1})
    print("[3] run_name ... OK")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frames_dir, splits_dir = make_dummy_dataset(tmp, videos_per_class=2, num_frames=8)
        base_cfg.update(frames_dir=str(frames_dir), splits_dir=str(splits_dir))
        grid = parse_grid(["lr=0.01,0.02"], base_cfg)

        # 4) sweep: one folder per run with its own config + checkpoints, sorted summary
        rows = sweep(base_cfg, grid, tmp / "runs")
        assert sorted(r["name"] for r in rows) == ["lr0.01", "lr0.02"], f"bad rows {rows}"
        accs = [r["val_accuracy"] for r in rows]
        assert accs == sorted(accs, reverse=True), f"summary not sorted: {accs}"
        for lr in (0.01, 0.02):
            run_dir = tmp / "runs" / f"lr{lr}"
            saved_cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
            assert saved_cfg["lr"] == lr and saved_cfg["checkpoint_dir"] == str(run_dir), (
                f"{run_dir}: wrong saved config")
            assert (run_dir / "best.pt").is_file() and (run_dir / "last.pt").is_file(), (
                f"{run_dir}: checkpoints missing")
        with open(tmp / "runs" / "summary.csv", newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        assert [r["name"] for r in csv_rows] == [r["name"] for r in rows], "CSV != returned rows"
        print("[4] sweep folders + summary ... OK")

        # 5) running again skips finished runs (checkpoints are not rewritten)
        mtime = (tmp / "runs" / "lr0.01" / "best.pt").stat().st_mtime_ns
        again = sweep(base_cfg, grid, tmp / "runs")
        assert again == rows, "resumed sweep gave different results"
        assert (tmp / "runs" / "lr0.01" / "best.pt").stat().st_mtime_ns == mtime, (
            "a finished run was trained again")
        print("[5] resume skips finished runs ... OK")

    print("All 5 checks passed.")


def main():
    """Parse the command line and run the sweep (or the self-test).

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Hyperparameter sweep over config values")
    parser.add_argument("--config", default="config.yaml", help="base YAML config")
    parser.add_argument("--grid", nargs="+", default=DEFAULT_GRID,
                        help="values to try, as key=v1,v2 (default: %(default)s)")
    parser.add_argument("--runs-dir", default="runs", help="folder for the run folders")
    parser.add_argument("--self-test", action="store_true", help="run the self-test and exit")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    base_cfg = load_config(args.config)
    sweep(base_cfg, parse_grid(args.grid, base_cfg), args.runs_dir)


if __name__ == "__main__":
    main()
