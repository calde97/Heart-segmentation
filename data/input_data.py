import nrrd
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data.preprocessing import read_single_dicom


class ImageSegmentationDataset(Dataset):
    def __init__(self, csv_file, transform=None, limit_for_testing=None, apply_hu_transformation=True,
                 apply_windowing=True):
        df = pd.read_csv(csv_file)
        self.image_paths = df['X_path'].values.tolist()
        self.mask_paths = df['y_path'].values.tolist()
        if limit_for_testing:
            self.image_paths = self.image_paths[:limit_for_testing]
            self.mask_paths = self.mask_paths[:limit_for_testing]
        self.transform = transform
        self.hu_transform_flag = apply_hu_transformation
        self.windowing_flag = apply_windowing

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_math = self.mask_paths[idx]

        image = read_single_dicom(image_path, hu_transformation_flag=self.hu_transform_flag,
                                  windowing_flag=self.windowing_flag)
        mask = nrrd.read(mask_math)[0]
        image = np.float32(image)
        # get max of image
        max_value = np.max(image)
        # get min of image
        min_value = np.min(image)
        # normalize image
        image = (image - min_value) / (max_value - min_value)
        #mask = np.float32(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = mask.float()

        # asser mask contains only 0 and 1
        assert np.all(np.unique(mask) == np.array([0., 1.]))

        return image, mask
