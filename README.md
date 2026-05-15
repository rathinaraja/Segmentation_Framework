# Semantic Segmentation Framework

A scalable, config-driven framework for training and evaluating semantic segmentation models.
Adding a new model requires **no changes** to `train.py`, `test.py`, or any utility.

---

## Project Structure

```
seg_framework/
├── configs/						       ← Input, output, model, and dataset specific hyper-parameter setting
│   ├── model1.yaml             
│   └── model2.yaml             
├── datasets/
│   ├── images/                ← Input images (.jpg .png .tif)
│   └── ground_truths/         ← Masks with matching filenames
├── logs/							         ← Results and models saved location
│   └── <model1>_<dataset>/
│       ├── Time_stamp1
│       		├── fold_1
│       			  ├── checkpoints
│       			  ├── metrics.csv
│       			  ├── model_fold1.log
│       			  ├── test_results.csv
│       		├── fold_2
│       		├── fold_3
│       		├── fold_4
│       		├── fold_5
│       		├── summary.csv   ← Per-epoch metrics across folds
│       ├── Time_stamp2 
│   └── <model2>_<dataset>/
├── modules/
│   ├── __init__.py            ← MODEL_REGISTRY + get_model()
│   ├── model1/
│       ├── model.py 
│   ├── model2/
│       ├── model.py  
├── utils/
│   ├── __init__.py 
│   ├── augmentations.py       ← Joint image+mask augmentations          
│   ├── config.py              ← load_config(), ConfigDict, validation
│   ├── dataset.py             ← SegmentationDataset + build_dataloaders()
│   ├── logger.py              ← Console/file logger + CSVLogger
│   ├── metrics.py             ← IoU, Dice, Pixel Accuracy, MetricTracker
│   └── train_utils.py         ← Loss function definition and other training related supporting utilities
├── infer.py
├── test.py
├── train.py
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install torch torchvision pyyaml pillow numpy
```

pip install -r requirements.txt

### 2. Organise data
```
datasets/images/          → image001.png, image002.png, ...
datasets/ground_truths/   → image001.png, image002.png, ...  (same filenames)
```
Masks should contain integer class indices (0, 1, 2, ...) as pixel values.

### 3. Configure
Edit `configs/unet.yaml` — key fields:

| Field | Description |
|---|---|
| `model.n_classes` | Number of segmentation classes |
| `model.n_channels` | Input channels (3 = RGB) |
| `training.loss` | `cross_entropy`, `dice`, or `dice_ce` |
| `training.learning_rate` | Initial LR |
| `dataset.augment` | `true` to enable joint augmentations |
| `logging.log_dir` | Where logs and checkpoints are saved |

### 4. Train
```bash
# UNet
python train.py --config configs/unet.yaml

# SegNet
python train.py --config configs/segnet.yaml

# Resume from checkpoint
python train.py --config configs/unet.yaml --resume logs/unet_dataset/checkpoints/best_model.pth

# Specify GPU
python train.py --config configs/unet.yaml --device cuda:1
```

### 5. Evaluate
```bash
# Reports Pixel Accuracy, Mean IoU, Dice on validation split
python test.py --config configs/unet.yaml \
               --checkpoint logs/unet_dataset/checkpoints/best_model.pth

# Save predicted mask PNGs
python test.py --config configs/unet.yaml \
               --checkpoint logs/unet_dataset/checkpoints/best_model.pth \
               --save_preds --output_dir outputs/predictions

# Inference only (no ground-truth needed)
python test.py --config configs/unet.yaml \
               --checkpoint logs/unet_dataset/checkpoints/best_model.pth \
               --images_dir /path/to/test/images \
               --save_preds
```

---

## Supported Options

### Loss functions (`training.loss`)
| Value | Description |
|---|---|
| `cross_entropy` | Standard pixel-wise CE |
| `dice` | Soft Dice loss |
| `dice_ce` | Dice + Cross-Entropy (recommended for imbalanced classes) |

### Optimizers (`optimizer.name`)
| Value | Notes |
|---|---|
| `adam` | Default; good general choice |
| `adamw` | Adam with decoupled weight decay |
| `sgd` | Needs `optimizer.momentum`; often better final accuracy |

### Schedulers (`scheduler.name`)
| Value | Notes |
|---|---|
| `cosine` | CosineAnnealingLR; use with Adam |
| `step` | StepLR; use `step_size` and `gamma` |
| `plateau` | ReduceLROnPlateau; good with SGD |

---

## Adding a New Model

Only 5 steps — `train.py` and `test.py` need **zero changes**.

```
Step 1 — modules/<model>/<model>_model.py    Define MyModel(nn.Module)
Step 2 — modules/<model>/<model>_parts.py   Building blocks (if needed)
Step 3 — modules/__init__.py                Add "mymodel": MyModel to MODEL_REGISTRY
Step 4 — process/<model>/<model>.py         class MyModelProcess(UNetProcess): pass
                                            (override methods only if needed)
Step 5 — process/__init__.py               Add "mymodel": MyModelProcess to PROCESS_REGISTRY
Step 6 — configs/<model>.yaml              Copy unet.yaml, set model.name: mymodel
```

Then simply run:
```bash
python train.py --config configs/mymodel.yaml
```

---

## Augmentations

Enabled per-config with `dataset.augment: true`. Applied only to the training split.

| Transform | Parameter |
|---|---|
| Horizontal flip | `p=0.5` |
| Vertical flip | `p=0.3` |
| Random rotation ±15° | `p=0.4` |
| Random crop + resize | `scale=(0.75,1.0), p=0.4` |
| Color jitter | `brightness/contrast/saturation/hue` |
| Gaussian blur | `radius=1.0, p=0.2` |

Custom augmentations can be added to `utils/augmentations.py` by subclassing `JointTransform`.

---

## Output Files

| File | Description |
|---|---|
| `logs/.../metrics.csv` | Epoch-level train/val metrics — ready to plot |
| `logs/.../best_model.pth` | Best checkpoint (lowest val loss) |
| `logs/.../*.log` | Timestamped training log |
| `outputs/predictions/*.png` | Predicted mask PNGs (if `--save_preds`) |
