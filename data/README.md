# Image Dataset Creation and Preprocessing

This repository contains all the necessary files for creating a custom dataset from input images, generating a CSV file, applying custom augmentations, and preprocessing the images and masks. It is designed to be compatible with both PyTorch and TensorFlow frameworks.

## Folder Structure

```
├── constants.py
├── create_dataset_from_csv_data.py
├── create_csv_data.py
├── custom_augmentations.py
├── input_data.py
├── preprocessing.py
└── README.md
```


## Files Description

- `constants.py`: Contains constants used across all the files in this folder.
- `create_dataset_from_csv_data.py`: Contains code for creating the dataset from the CSV file.
- `create_csv_data.py`: Contains code for creating a CSV file for the dataset by reading all the paths of the images and masks.
- `custom_augmentations.py`: Contains code for creating custom augmentations for the dataset.
- `input_data.py`: Contains code for reading the data and creating custom datasets for PyTorch and TensorFlow.
- `preprocessing.py`: Contains code for preprocessing the images and masks, and utility functions called by other files in this folder.

## Usage

1. Make sure you have all the required dependencies installed in your environment.
2. Update the `constants.py` file with the appropriate paths and settings for your dataset.
3. Run `create_csv_data.py` to generate the CSV file containing the paths of images and masks.
4. Use `create_dataset_from_csv_data.py` to create the dataset from the CSV file.
5. Apply custom augmentations using the `custom_augmentations.py` file.
6. Utilize `input_data.py` to read the data and create custom datasets for PyTorch and TensorFlow.
7. Perform preprocessing on the images and masks using the `preprocessing.py` file.

