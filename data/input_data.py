import nrrd
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data.preprocessing import read_single_dicom


class ImageSegmentationDataset(Dataset):
    def __init__(self, csv_file, transform=None, limit_for_testing=None):
        df = pd.read_csv(csv_file)
        self.image_paths = df['X_path'].values.tolist()
        self.mask_paths = df['y_path'].values.tolist()
        if limit_for_testing:
            self.image_paths = self.image_paths[:limit_for_testing]
            self.mask_paths = self.mask_paths[:limit_for_testing]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_math = self.mask_paths[idx]

        image = read_single_dicom(image_path)
        mask = nrrd.read(mask_math)[0]
        image = np.float32(image)
        # get max of image
        max_value = np.max(image)
        # get min of image
        min_value = np.min(image)
        # normalize image
        image = (image - min_value) / (max_value - min_value)
        mask = np.float32(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
