# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based deep learning project for **multi-class weather image classification**, developed for the **2026 睿抗机器人开发者大赛 (CAIP)** intelligent algorithm competition. It classifies images into 8 categories: `cloudy`, `haze`, `rainy`, `shine`, `snow`, `sunny`, `sunrise`, `thunder`.

The main code lives in `github/weather-recognition/`. The project is a fork/clone of an existing weather-recognition repository, adapted for the competition.

## Common Commands

All commands should be run from the `github/weather-recognition/` directory unless otherwise noted.

### Training
```bash
cd github/weather-recognition
python train.py
```
- Training logs are written to `./log/<timestamp>/` for TensorBoard.
- Models are saved to `./model/` when validation accuracy exceeds the threshold (default `0.75`) and at the end of training.
- The script uses `Adam` optimizer, `CrossEntropyLoss`, and runs for 40 epochs by default (see `config.py`).

### Inference / Prediction
```bash
cd github/weather-recognition
python pridect.py
```
- Note the filename is `pridect.py` (typo in original code).
- Hardcoded paths in `if __name__ == '__main__'` will likely need adjustment before running.

### View Training Logs
```bash
tensorboard --logdir=github/weather-recognition/log
```

## Environment & Dependencies

- **Python**: 3.10+ (managed via Conda as configured in `.vscode/settings.json`).
- **PyTorch**: `torch` and `torchvision` are required.
- **Other key dependencies**: `Pillow`, `matplotlib`, `tensorboard`, `numpy`.
- There is no `requirements.txt` or `pyproject.toml`; install dependencies manually via Conda/Pip as needed.
- **Windows-specific**: `num_workers` in `config.py` is set to `0` to avoid multiprocessing errors on Windows.

## High-Level Architecture

### Data Pipeline
- `data_loader.py` defines a custom `WeatherDataSet` that loads **all images into memory at import time** (side effect on module load). It expects images organized in subdirectories by class name under `Common.basePath`.
- The dataset is split 80/20 into train/validation using `torch.utils.data.random_split`.
- `DataLoader` instances (`trainLoader`, `valLoader`) are also created at module import time.
- Labels are one-hot encoded manually in `data_loader.py` (8-class float tensor), even though `CrossEntropyLoss` expects integer class indices.

### Model Architecture (`model.py`)
- Built on top of `torchvision.models.resnet50` with pretrained weights loaded from a **local file**: `./model/resnet50-11ad3fa6.pth`.
- A custom `WeatherModel` wraps ResNet-50 and adds: `ReLU` → `Dropout(0.1)` → `Linear(1000, 8)` → `Softmax(dim=1)`.

### Training Loop (`train.py`)
- Imports `model`, `trainLoader`, and `valLoader` at the top level, triggering the data loading side effects immediately.
- Uses TensorBoard `SummaryWriter` for logging train/validation loss and accuracy.
- Saves the **entire model** (not just `state_dict`) when validation accuracy improves.

## Important Configuration & Paths

- **`config.py:Common.basePath`**: Hardcoded to `D:/Data/weather/source/all/`. This path likely does not exist on the current machine. The actual extracted dataset is located at `data/RSCM/classification/weather_classification/` in the project root.
- **`config.py:Common.labels`**: 8 classes. The primary RSCM dataset only provides 6 of these (`cloudy`, `haze`, `rainy`, `snow`, `sunny`, `thunder`). `shine` and `sunrise` come from the smaller MWD dataset.
- **Model weights path**: `./model/resnet50-11ad3fa6.pth` is required for the model to initialize. The final trained model is saved to `./model/weather-<timestamp>.pth`.

## Dataset Layout

The project includes three datasets under `data/`:

- **`data/RSCM/`** (60,000 images, 6 classes, 1.9 GB): Primary dataset. Extracted to `data/RSCM/classification/weather_classification/`.
- **`data/MWD/`** (1,125 images, 4 classes, 91 MB): Smaller dataset containing `cloudy`, `rain`, `shine`, `sunrise`.
- **`data/WEAPD/`** (6,862 images, 11 classes, 592 MB): Not yet extracted (`WEAPD_dataset.rar`).

## Known Issues & Quirks

1. **Softmax + CrossEntropyLoss mismatch**: `model.py` applies `Softmax` as the final layer, but `train.py` uses `nn.CrossEntropyLoss`, which internally applies `log_softmax`. Applying `Softmax` before `CrossEntropyLoss` is mathematically incorrect and can hurt training stability.
2. **One-hot labels with CrossEntropyLoss**: `data_loader.py` produces one-hot float labels, but `CrossEntropyLoss` expects integer class indices. The code happens to work because `argmax` is used on the labels before comparison, but the loss computation itself may behave unexpectedly with one-hot floats.
3. **Module-level side effects**: Importing `data_loader.py` or `train.py` immediately loads all dataset images into RAM and splits them. Be cautious when importing these modules in scripts or REPLs.
4. **Hardcoded inference paths**: `pridect.py` contains hardcoded image and model paths in its `__main__` block.
