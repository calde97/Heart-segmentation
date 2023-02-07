import os
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import numpy as np
import torchvision.transforms as T
import pandas as pd
import nrrd

from data.preprocessing import read_single_dicom
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageSegmentationDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        df = pd.read_csv(csv_file)
        self.image_paths = df['X_path'].values.tolist()
        self.mask_paths = df['y_path'].values.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_math = self.mask_paths[idx]

        image = read_single_dicom(image_path)
        mask = nrrd.read(mask_math)[0]

        image = np.float32(image)
        mask = np.float32(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


if __name__ == '__main__':

    df = pd.read_csv('data/csv_files/train.csv')
    # filter for the patient AM08
    df = df[df['patient'] == 'HCM04']
    # get values of column X_path
    X_path = df['X_path'].values
    y_path = df['y_path'].values

    images = [read_single_dicom(path) for path in X_path]
    images_without_preprocessing = [read_single_dicom(path, hu_transformation_flag=False, windowing_flag=False) for path
                                    in
                                    X_path]
    masks = [nrrd.read(path)[0] for path in y_path]
    # from df get all the X-paths in a list and all the y-paths in a list
    X_path = df['X_path'].values.tolist()
    y_path = df['y_path'].values.tolist()

    image = images[0]

    # transform list of images to numpy array
    images = np.array(images)

    # expand image to 3 channels
    image = np.expand_dims(image, axis=2)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])


    dataset = ImageSegmentationDataset(csv_file='data/csv_files/test.csv', transform=None)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap=plt.cm.bone)
    plt.imshow(label, alpha=0.2)
    plt.show()

    # plot all 4 the train features and the train labels
    fig, ax = plt.subplots(4, 1, figsize=(10, 10))
    for i in range(4):
        ax[i].imshow(train_features[i].squeeze(), cmap=plt.cm.bone)
        ax[i].imshow(train_labels[i].squeeze(), alpha=0.2)
    plt.show()

