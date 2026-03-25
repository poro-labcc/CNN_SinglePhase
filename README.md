# Deep Learning Surrogate for Lattice Boltzmann Method (LBM) Simulations

## Overview
This repository provides a PyTorch-based training pipeline for developing deep learning surrogates of Lattice Boltzmann Method (LBM) simulations.

---
# About the models



---
# Training Process


## Usage and Execution

The training pipeline is controlled via command-line arguments and reads hyperparameters from structured `.json` configuration files.

### 1. Standard Training

To start a new training process, provide a JSON configuration file.  
If no file is specified, the script defaults to `config.json` in the root directory.

**Example:**
```bash
python main_TrainModel.py --config experiment_01.json
```

---

### 2. Resuming an Experiment

To resume a previous training session or reproduce an experiment, provide the target results directory.  

This feature is particularly useful for:
- Splitting long training runs into multiple executions  
- Protecting against system interruptions or crashes  
- Implementing curriculum learning strategies  

The script automatically loads the `metadata.json` generated during the original run and ignores any `--config` file.

**Example:**
```bash
python main_TrainModel.py --folder ../NN_Results/NN_Trainning_23_March_2026_01-46PM
```

---

## Configuration Parameters

The `config.json` file controls all aspects of the model, dataset handling, and training process.

### General Structure

```json
{
    "model_name": "danny_z",
    "binary_input": true,
    "NN_dataset_folder": "../NN_Datasets/",
    "dataset_train_name": "Train_Dataset.h5",
    "dataset_valid_name": "Valid_Dataset.h5",
    "train_range": [0, 8],
    "valid_range": [0, 2],
    "batch_size": 8,
    "num_workers": 4,
    "num_threads": 18,
    "N_epochs": 100,
    "partial_epochs": 100,
    "patience": 50,
    "learning_rate": 0.0006,
    "earlyStopping_loss": "PRPE",
    "backPropagation_loss": "Corr_MSE",
    "optimizer": "ADAM",
    "weight_init": null,
    "seed": 42,
    "train_comment": "Description of the current experiment."
}
```

---

### Parameter Description

- **`model_name`**  
  Specifies the neural network architecture. Available options:  
  `'javier_z'`, `'danny_z'`, `'danny_y'`, `'danny_x'`, `'danny_zyxp'`.

- **`binary_input`**  
  Defines the input representation:
  - `true`: binary solid/void geometry  
  - `false`: distance transform or continuous representation  

- **`NN_dataset_folder`**  
  Directory containing the dataset files.

- **`dataset_train_name`**  
  Training dataset filename (must exist inside `NN_dataset_folder`).

- **`dataset_valid_name`**  
  Validation dataset filename (must exist inside `NN_dataset_folder`).

- **`train_range`**  
  Index range used from the training dataset.  
  If `null`, the full dataset is used.

- **`valid_range`**  
  Index range used from the validation dataset.  
  If `null`, the full dataset is used.

- **`batch_size`**  
  Number of samples per batch (i.e., per weight update).

- **`N_epochs`**  
  Maximum number of training epochs.

- **`partial_epochs`**  
  Number of epochs executed per run.  
  Enables splitting long training jobs into multiple executions.

- **`patience`**  
  Early stopping threshold. Training stops if no improvement in the monitored metric occurs for this number of epochs.

- **`learning_rate`**  
  Learning rate used by the optimizer.

- **`optimizer`**  
  Optimization algorithm. Supported options:
  - `"ADAM"`
  - `"ADAMW"`
  - `"SGD"`

- **`backPropagation_loss`**  
  Loss function used to compute gradients and update model weights.

- **`earlyStopping_loss`**  
  Metric used to track validation performance and determine the best model.

- **`weight_init`**  
  Weight initialization strategy. Options:
  - `null` (default initialization)
  - `"xavier"`
  - `"he"`
  - `"zeros"`

- **`seed`**  
  Random seed for reproducibility across runs.

- **`train_comment`**  
  Free-text description of the experiment. Stored for tracking and reproducibility.

---

## Data Handling

### Lazy Loading

Datasets must be provided in `.h5` (HDF5) format.

The `LazyDatasetTorch` class performs on-the-fly data loading to minimize RAM usage, making it suitable for large-scale datasets. Instead of preloading all data into memory, batches are dynamically loaded from disk during training.

Solid regions (where velocity and pressure are strictly zero) are preserved using a binary geometric mask. This prevents normalization artifacts from distorting physical boundaries during convolutional operations.

---

### Custom Datasets

Custom dataset classes can be used if needed.

To implement a different dataset pipeline, modify the dataset object definition in:

```bash
main_TrainModel.py
```

---
# Validation Process

## Quantitative analysis


## Qualitative analysis
