# EMG-Based Human Pose Forecasting

This project aims to build a neural network model that predicts human hand poses based on EMG signals, using Python and PyTorch.

## Dataset

We use the open-source **Meta EMG2Pose** dataset, which includes EMG and motion capture data. However, the official META dataset is currently too large to test on, so I am using a custom "fake" dataset generated using OpenAI.

## Project Structure

- `data/` - Contains raw HDF5 files and `metadata.csv`
- `models/` - Custom PyTorch model definitions
- `scripts/` - Training and evaluation scripts
- `notebooks/` - Jupyter notebooks for EDA and model testing
- `utils/` - Data loader and preprocessing tools
- `config/` - Configuration files (YAML/JSON)

## Getting Started

### Setup

```bash
pip install -r requirements.txt
```

### Training

```bash
python -m scripts.train_with_logging
```

### Project Goals

- Build a CNN model from scratch for EMG to Pose estimation
- Perform cross-validation across users and stages
- Extend to real-time pose prediction

