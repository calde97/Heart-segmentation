import nrrd
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from data.preprocessing import read_single_dicom


def normalize(image):
    # get max of image
    max_value = torch.max(image)
    # get min of image
    min_value = torch.min(image)
    # normalize image
    image = (image - min_value) / (max_value - min_value)
    return image


class ImageSegmentationDataset(Dataset):
    def __init__(self, csv_file, transform=None, limit_for_testing=None, apply_hu_transformation=True,
                 apply_windowing=True, starting_index=0):
        df = pd.read_csv(csv_file)
        self.image_paths = df['X_path'].values.tolist()
        self.mask_paths = df['y_path'].values.tolist()
        self.patients = df['patient'].values.tolist()
        self.serial_numbers = df['serial_number'].values.tolist()
        if limit_for_testing:
            self.image_paths = self.image_paths[starting_index:limit_for_testing]
            self.mask_paths = self.mask_paths[starting_index:limit_for_testing]
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
        # mask = np.float32(mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        mask = mask.float()
        mask = (mask >= 0.5).float()
        mask = mask.unsqueeze(0)
        # asser mask contains only 0 and 1 and if raise print the unique values

        assert (mask == 0.).sum() + (mask == 1.).sum() == mask.numel(), mask.unique()

        image = normalize(image)

        return image, mask, self.patients[idx], self.serial_numbers[idx]


class ImageSegmentationMultipleSlicesAsChannelsDataset(Dataset):
    def __init__(self, csv_file, transform=None, limit_for_testing=None, apply_hu_transformation=True,
                 apply_windowing=True, starting_index=0, slices=5):
        df = pd.read_csv(csv_file)
        self.image_paths = df['X_path'].values.tolist()
        self.mask_paths = df['y_path'].values.tolist()
        self.patients = df['patient'].values.tolist()
        self.serial_numbers = df['serial_number'].values.tolist()
        if limit_for_testing:
            self.image_paths = self.image_paths[starting_index:limit_for_testing]
            self.mask_paths = self.mask_paths[starting_index:limit_for_testing]
        self.transform = transform
        self.hu_transform_flag = apply_hu_transformation
        self.windowing_flag = apply_windowing
        self.slices = slices
        self.len_dataset = len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        indices = self.get_all_slices_and_masks(idx)
        image_paths = [self.image_paths[index] for index in indices]
        mask_paths = [self.mask_paths[index] for index in indices]

        images = [read_single_dicom(image_path, hu_transformation_flag=self.hu_transform_flag,
                                  windowing_flag=self.windowing_flag) for image_path in image_paths]
        masks = [nrrd.read(mask_math)[0] for mask_math in mask_paths]

        images = [np.float32(image) for image in images]
        # mask = np.float32(mask)

        if self.transform:
            transformed_list = [self.transform(image=image, mask=mask) for image, mask in zip(images, masks)]
            images = [transformed['image'] for transformed in transformed_list]
            masks = [transformed['mask'] for transformed in transformed_list]

        mask = masks[0]
        mask = mask.float()
        mask = (mask >= 0.5).float()
        mask = mask.unsqueeze(0)
        # asser mask contains only 0 and 1 and if raise print the unique values

        assert (mask == 0.).sum() + (mask == 1.).sum() == mask.numel(), mask.unique()

        images = [normalize(image) for image in images]
        # concatenate images as channels
        image = torch.cat(images, dim=0)

        return image, mask, self.patients[idx], self.serial_numbers[idx]

    def get_all_slices_and_masks(self, idx):
        indices = [index for index in range(idx, idx + self.slices)]
        # if some of the indices are out of range, then we change them to the last index
        indices = [index if index < self.len_dataset else self.len_dataset - 1 for index in indices]
        correct_patient = self.patients[idx]
        # loop through patients[indices] and check if they are the same as the patient of the first index
        # if not, then we change the index to the last index
        different_patient = -1
        for index in indices:
            if self.patients[index] != correct_patient:
                different_patient = index
                break
        if different_patient != -1:
            indices = [index if index < different_patient else different_patient - 1 for index in indices]

        return indices



class ImageSegmentationLSTM(Dataset):
    def __init__(self, csv_file, transform=None, limit_for_testing=None, apply_hu_transformation=True,
                 apply_windowing=True, starting_index=0, slices=5):
        df = pd.read_csv(csv_file)
        self.image_paths = df['X_path'].values.tolist()
        self.mask_paths = df['y_path'].values.tolist()
        self.patients = df['patient'].values.tolist()
        self.serial_numbers = df['serial_number'].values.tolist()
        if limit_for_testing:
            self.image_paths = self.image_paths[starting_index:limit_for_testing]
            self.mask_paths = self.mask_paths[starting_index:limit_for_testing]
        self.transform = transform
        self.hu_transform_flag = apply_hu_transformation
        self.windowing_flag = apply_windowing
        self.slices = slices
        self.len_dataset = len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        indices = self.get_all_slices_and_masks(idx)
        image_paths = [self.image_paths[index] for index in indices]
        mask_paths = [self.mask_paths[index] for index in indices]

        images = [read_single_dicom(image_path, hu_transformation_flag=self.hu_transform_flag,
                                  windowing_flag=self.windowing_flag) for image_path in image_paths]
        masks = [nrrd.read(mask_math)[0] for mask_math in mask_paths]

        images = [np.float32(image) for image in images]
        # mask = np.float32(mask)

        if self.transform:
            transformed_list = [self.transform(image=image, mask=mask) for image, mask in zip(images, masks)]
            images = [transformed['image'] for transformed in transformed_list]
            masks = [transformed['mask'] for transformed in transformed_list]

        mask = masks[0]
        mask = mask.float()
        mask = (mask >= 0.5).float()
        mask = mask.unsqueeze(0)
        # asser mask contains only 0 and 1 and if raise print the unique values

        assert (mask == 0.).sum() + (mask == 1.).sum() == mask.numel(), mask.unique()

        images = [normalize(image) for image in images]
        # concatenate images as channels
        image = torch.cat(images, dim=0)

        return image, mask, self.patients[idx], self.serial_numbers[idx]

    def get_all_slices_and_masks(self, idx):

        starting_index = idx + self.slices - 1

        if starting_index >= self.len_dataset:
            starting_index = self.len_dataset - 1


        indices = [index for index in range(starting_index,  -1, -1)]

        # if some of the indices are out of range, then we change them to the last index
        #indices = [index if index < self.len_dataset else self.len_dataset - 1 for index in indices]
        correct_patient = self.patients[starting_index]
        # loop through patients[indices] and check if they are the same as the patient of the first index
        # if not, then we change the index to the last index
        different_patient = -1
        for index in indices:
            if self.patients[index] != correct_patient:
                different_patient = index
                break
        if different_patient != -1:
            indices = [index if index > different_patient else different_patient + 1 for index in indices]

        return indices
