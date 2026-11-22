# Real-time violence detection with TSM + MobileNetV3

A MobileNetV3 image backbone (ImageNet-pretrained) with the Temporal Shift Module (TSM)
inserted into its residual blocks. It classifies short video clips as violence / non-violence
and runs in real time.

## Setup

```
pip install -r requirements.txt
```

## Data

Put the raw videos in `data/raw` (set by `raw_dir` in `config.yaml`) in one of two layouts.
The class folder names must match `classes` in the config (default `[NonFight, Fight]`).

```
A) split folders (e.g. RWF-2000)       B) class folders only (e.g. RLVS)
   data/raw/train/Fight/*.avi             data/raw/Fight/*.mp4
   data/raw/train/NonFight/*.avi          data/raw/NonFight/*.mp4
   data/raw/val/Fight/*.avi
   data/raw/val/NonFight/*.avi
```

- **Layout A:** if there is no `test` folder, the official `val` folder is used as the test
  split, and a new validation split (`val_ratio`) is taken from train.
- **Layout B:** test (`test_ratio`) and validation (`val_ratio`) are taken at random.

All random splits are stratified by class and seeded.

## Run

| Step | Command | Output |
|---|---|---|
| 1. Extract frames + make splits (once) | `python prepare_data.py` | `data/frames/`, `data/splits/{train,val,test}.csv` |
| 2. Train | `python train.py` | `checkpoints/last.pt`, `checkpoints/best.pt` (best val accuracy) |
| 2b. Or sweep hyperparameters | `python sweep.py --grid lr=0.003,0.01,0.03 weight_decay=0.0001,0.0005` | `runs/<name>/` per run, `runs/summary.csv` (sorted by val accuracy) |
| 3. Test | `python test.py` | printed report + `results/test_metrics.json` |
| 4. Live demo | `python infer.py --source video.mp4` or `--source 0` (webcam) | window with the label; `q` quits |
| Speed | `python benchmark.py` | latency per clip, clips/s, frames/s |

All settings live in `config.yaml`. Every script takes `--config other.yaml`.

After a sweep, pick the best run by its **validation** score, then test only that one:
`python test.py --checkpoint runs/<name>/best.pt`. Choosing by test score would make the
test result optimistic. Re-running the same sweep command skips finished runs, so an
interrupted sweep can be resumed.
`test.py` and `infer.py` also take `--checkpoint path.pt` (default: `checkpoints/best.pt`).

## Files

| File | Contents |
|---|---|
| `tsm_mobilenetv3.py` | Model: MobileNetV3 (torchvision-compatible weights) + temporal shift |
| `dataset.py` | `VideoDataset`: TSN-style frame sampling, crop / flip, normalization |
| `prepare_data.py` | Frame extraction and train / val / test splitting |
| `train.py` | Training loop (SGD + cosine LR, mixed precision on GPU), checkpoints |
| `sweep.py` | Hyperparameter sweep: one training run per combination, summary table |
| `test.py` | Accuracy, precision / recall / F1 (positive class = `classes[1]`), confusion matrix |
| `infer.py` | Sliding-window real-time inference on a video or webcam |
| `benchmark.py` | Speed measurement |
| `utils.py` | Shared code: config, seed, device, model / checkpoint loading, metrics |
| `config.yaml` | All paths and hyperparameters |

## Self-tests

Every file checks itself on tiny fake data, with no dataset or download needed:

```
python tsm_mobilenetv3.py
python utils.py
python dataset.py
python prepare_data.py --self-test
python train.py --self-test
python sweep.py --self-test
python test.py --self-test
python infer.py --self-test
python benchmark.py --self-test
```

## Sources

- **TSM:** Lin, Gan, Han, "TSM: Temporal Shift Module for Efficient Video Understanding",
  ICCV 2019 (arXiv:1811.08383). Code: https://github.com/mit-han-lab/temporal-shift-module
- **MobileNetV3:** Howard et al., "Searching for MobileNetV3", ICCV 2019 (arXiv:1905.02244).
  Code: torchvision `models/mobilenetv3.py`
- **TSN** (frame sampling): Wang et al., "Temporal Segment Networks", ECCV 2016 (arXiv:1608.00859)
- **SE blocks:** Hu, Shen, Sun, "Squeeze-and-Excitation Networks", CVPR 2018 (arXiv:1709.01507)
