# PALNet

## 1. Overview

PALNet is a point-supervised framework for individual peanut seedling localization and stand establishment assessment from UAV imagery using Gaussian heatmap regression.

This repository provides a 200-sample public dataset for reproducing the PALNet workflow and testing the released code. It is intended for workflow reproduction and code testing and does not claim to reproduce every reported experiment.

## 2. Method

The released model uses a **ResNet50 + FPN + ECA + ConfigurableUNet** architecture. Point annotations are converted to Gaussian heatmaps for supervised regression. Predicted heatmaps are processed to recover seedling coordinates.

The training implementation uses the CombinedLoss loss function and the Adam optimizer, with mixed-precision training, gradient clipping, validation, best-checkpoint selection, and test evaluation.

## 3. Public dataset

The public release contains:

- 200 TIFF image blocks
- 200 corresponding LabelMe JSON annotation files
- 3,469 point annotations
- Image size: 640 × 640 pixels
- LabelMe version: 5.5.0
- `shape_type`: `point`
- `label`: `peanut`

Images and annotations correspond by the same file stem.

## 4. Dataset split

The provided CSV contains the sample assignments used by this public release: peanut2024_split.csv.

- Train: 120
- Validation: 40
- Test: 40
- Total: 200

The CSV contains three columns: image, annotation, and split. The code reads these assignments directly rather than generating a new random split.

## 5. Repository structure

The current GitHub-facing directory contains:

```text
PALNet/
├── PALNet.py
├── README.md
├── requirements.txt
├── peanut2024-split.csv
└── PALNet原始代码.py  (historical backup; not part of the formal release)
```

The formal release consists of PALNet.py, README.md, requirements.txt, and the split CSV. The historical backup is retained locally for reference and should not be included in the formal release.

When the public data are available locally, organize them as:

```text
Peanut-2024-public/
├── images/
│   └── 200 TIFF files
└── annotations/
    └── 200 LabelMe JSON files
```

The split CSV is kept at the repository root. Each JSON imagePath uses the relative form:

```text
../images/<image_filename>.tif
```

## 6. Installation

Install the required packages with:

```bash
pip install -r requirements.txt
```

The repository does not prescribe a specific Python or CUDA version.

## 7. Training

Run the workflow with paths to the public data and the split CSV at the repository root:

```bash
python PALNet.py \
    --image-dir ./Peanut-2024-public/images \
    --annotation-dir ./Peanut-2024-public/annotations \
    --split-csv ./peanut2024-split.csv \
    --output-dir ./outputs
```

The command-line arguments are:

- `--image-dir`: TIFF image directory
- `--annotation-dir`: LabelMe JSON annotation directory
- `--split-csv`: CSV containing the public split assignments
- `--output-dir`: directory for generated outputs

The training configuration is:

- Epochs: 100
- Batch size: 4
- Learning rate: `5e-4`
- Weight decay: `1e-4`
- Random seed: 42
- Gaussian sigma: 10 pixels
- Scheduler: `CosineAnnealingLR`
- `T_max`: 30
- `eta_min`: `1e-6`
- Optimizer: Adam
- Mixed-precision training: enabled
- Gradient clipping: enabled

## 8. Evaluation and outputs

Validation is performed during training. The best checkpoint is selected using the existing validation-loss logic. The script then evaluates the train and test assignments read from the split CSV.

Generated checkpoints, evaluation summaries, plots, and prediction visualizations are written under the selected `--output-dir`.

The script does not provide a separate evaluation-only command-line interface.

## 9. Reproducibility

Use the supplied split CSV directly. The code reads the train, validation, and test assignments from the CSV and does not generate a new random split. The random seed is 42.

## 10. Data and code availability

This repository provides the PALNet source code, runtime requirements, split definition, and documentation for the 200-sample public release. The public data are organized as paired TIFF images and LabelMe JSON point annotations when provided locally.

## 11. Citation

If you use this code or public dataset, please cite the associated PALNet study:

> Individual peanut seedling localization and stand establishment assessment from UAV imagery using point-supervised Gaussian heatmap regression

## 12. License

License: To be specified by the authors before public release.
