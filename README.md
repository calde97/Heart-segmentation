# Automatic left ventricle segmentation

This repository contains code for performing left ventricular segmentation using deep learning. The code provides all the necessary steps for training the model and performing inference on **CT scans**. Additionally, a detailed report on the model and techniques used will be provided in the future.

## Repository structure

The repository is structured as follows:

- `Data`: This folder contains code to load images from DICOM files and masks from NRRD files. It also includes code to create a CSV file for the training/validation test split. Various preprocessing steps are also included in this folder. Here there also the code for creating the custom dataset for pytorch and tensorflow.
- `Model`: This folder contains the DL model for left ventricular segmentation.
- `Training`: This folder contains code to train the models. It includes a fix for the IoU metrics.
- `Evaluation`: This folder contains code for evaluating the model's performance.
- `dash-visualizations`: This folder includes the Plotly visualization of the model results on the validation data.
- `LICENSE`: This file contains the repository's license.
- `requirements.txt`: This file lists the dependencies required to run the code in this repository.
- `model_metrics`: This folder contains the metrics for the model. It includes the loss and IoU metrics for the training and validation data.

## Getting started

To get started with this repository, you will need to install the required dependencies listed in `requirements.txt`. You can do this by running the following command:

```bash
pip install -r requirements.txt
```
Remember to also intall pytorch from the official website : https://pytorch.org/



Once you have installed the dependencies, you can use the code in this repository to train and evaluate the UNET model for left ventricular segmentation.

## Contributors

This repository was created by Juan Calderon
