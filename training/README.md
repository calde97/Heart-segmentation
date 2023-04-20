# README

This is a training script for image segmentation models. It provides functionalities for training different segmentation models, such as U-Net, U-Net with LSTM, and hybrid models, and it supports multiple input slices and LSTM-based input handling.

## Requirements

- Python 3.x
- PyTorch
- Torchmetrics
- PyYAML
- tqdm
- wandb (Weights & Biases)

## Usage

1. Make sure you have all the required libraries installed.
2. Configure the training settings in the corresponding YAML file (e.g., `yaml_files/unet-lstm.yaml`).
3. Run the script: `python generic_training.py`

## Features

- Training of different image segmentation models, such as U-Net, U-Net with LSTM, and hybrid models.
- Support for multiple input slices and LSTM-based input handling.
- Configurable training settings using YAML files.
- Data augmentation for training and validation datasets.
- Weights & Biases integration for experiment tracking and logging.

## Configuration

The training settings can be configured using YAML files. An example YAML file is provided (`yaml_files/unet-lstm.yaml`). The configuration file includes settings like batch size, learning rate, number of epochs, loss function, model, training and validation paths, etc.

## Training Process

The training process consists of the following steps:

1. Initialize the training class with the configuration settings.
2. Load and preprocess the training and validation datasets.
3. Train the selected model using the specified training settings.
4. Evaluate the model on the validation dataset and log the results.
5. Save the best model based on the validation loss.

The script also supports early stopping if there is no improvement in validation loss for a specified number of epochs.



# YAML Configuration File

This YAML file is used to configure the training settings for the `generic_training.py` script. Below is an explanation of each parameter:

```yaml
program: generic_training.py
method: random
metric:
  name: mean_val_loss
  goal: minimize
parameters:
  batch_size:
    values : [4,8] # List of batch sizes to choose from
  learning_rate:
    min: 0.00001 # Minimum learning rate
    max: 0.01    # Maximum learning rate
    distribution: uniform
  num_epochs:
    value: 300 # Number of training epochs
  criterion:
    values: ['dice_loss'] # Loss function used for training
  model:
    value: 'unet' # Model used for training (e.g., 'unet')
  training_path:
    value: "../data/csv_files/train-fix.csv" # Path to the training dataset
  validation_path:
    value: "../data/csv_files/val-fix.csv" # Path to the validation dataset
  max_non_improvement_epochs:
    value: 50 # Number of epochs without improvement before early stopping
  min_val_iou_for_saving:
    value: 0.20 # Minimum IoU value required to save the model
  dataset_as_slices:
    value: False # Set to True to use dataset as slices
  num_slices:
    value: -1 # Number of slices for the input (use -1 if not applicable)
  debug_testing:
    value: False # Set to True to enable debug testing mode
