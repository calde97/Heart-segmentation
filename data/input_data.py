import nrrd
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from data.preprocessing import read_single_dicom


import pandas as pd
import numpy as np
import tensorflow as tf



def normalize(image):
    # get max of image
    max_value = torch.max(image)
    # get min of image
    min_value = torch.min(image)
    # normalize image
    image = (image - min_value) / (max_value - min_value)
    return image


def normalize_numpy(image):
    # get max of image
    max_value = np.max(image)
    # get min of image
    min_value = np.min(image)
    # normalize image
    image = (image - min_value) / (max_value - min_value)
    return image



class ImageSegmentationDataset(Dataset):
    '''
    Dataset for image segmentation. It takes as input a csv file with the path of the images and the path of the masks.
    It takes as input also the transformations to apply to the images and the masks. It is compatible with pytorch
    '''
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
    '''
    Dataset for image segmentation. It takes as input a csv file with the path of the images and the path of the masks.
    It takes as input also the transformations to apply to the images and the masks. It is compatible with pytorch.
    It is the dataset used for the Unet2.5D model that takes as input multiple slices as channels.
    '''
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

    '''
    Dataset for image segmentation. It takes as input a csv file with the path of the images and the path of the masks.
    It takes as input also the transformations to apply to the images and the masks. It is compatible with pytorch.
    It is the dataset used for the LSTM model that takes as input multiple slices as a time series.
    '''
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
        return len(self.image_paths) - self.slices + 1

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

        masks = [mask.float() for mask in masks]
        masks = [(mask >= 0.5).float() for mask in masks]
        masks = [mask.unsqueeze(0) for mask in masks]
        mask = torch.cat(masks, dim=0)
        mask = mask.unsqueeze(1)
        # asser mask contains only 0 and 1 and if raise print the unique values

        assert (mask == 0.).sum() + (mask == 1.).sum() == mask.numel(), mask.unique()

        images = [normalize(image) for image in images]
        # concatenate images as channels
        image = torch.cat(images, dim=0)
        image = image.unsqueeze(1)

        return image, mask, self.patients[idx], self.serial_numbers[idx]

    def get_all_slices_and_masks(self, idx):

        starting_index = idx + self.slices - 1

        if starting_index >= self.len_dataset:
            starting_index = self.len_dataset - 1


        indices = [index for index in range(starting_index,  idx -1, -1)]



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




class ImageSegmentationDatasetTF(tf.keras.utils.Sequence):
    def __init__(self, csv_file, batch_size=4, transform=None, limit_for_testing=None, apply_hu_transformation=True,
                 apply_windowing=True, starting_index=0, height=256, width=256, slices=3):
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
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.slices = slices


    def __len__(self):
        return int(np.ceil(len(self.image_paths) / (self.batch_size * self.slices)))


    def __getitem__(self, idx):
        delta = self.slices * self.batch_size
        #indices = [i for i in range((idx+1) * delta - 1, idx * delta -1, -1)]
        #indices = self.check_indices(indices)


        batch_x_paths = self.image_paths[idx * delta:(idx + 1) * delta]
        #batch_x_paths = [self.image_paths[i] for i in indices]
        batch_y_paths = self.mask_paths[idx * delta:(idx + 1) * delta]
        batch_patients = self.patients[idx * delta:(idx + 1) * delta]

        #reverse the order of batch_x_paths
        batch_x_paths = batch_x_paths[::-1]
        batch_y_paths = batch_y_paths[::-1]
        batch_patients = batch_patients[::-1]

        batch_x_paths, batch_y_paths = self.adjust_indices(batch_x_paths, batch_y_paths, batch_patients)



        #batch_y_paths = [self.mask_paths[i] for i in indices]

        batch_x = np.zeros((self.batch_size, self.slices, self.height, self.width, 1))
        batch_y = np.zeros((self.batch_size, self.slices, self.height, self.width, 1))

        for i, (image_path, mask_path) in enumerate(zip(batch_x_paths, batch_y_paths)):
            image = read_single_dicom(image_path, hu_transformation_flag=self.hu_transform_flag,
                                      windowing_flag=self.windowing_flag)
            mask = nrrd.read(mask_path)[0]
            image = np.float32(image)

            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image, mask = transformed['image'], transformed['mask']

            mask = np.float32(mask)
            mask = (mask >= 0.5).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)

            assert (mask == 0.).sum() + (mask == 1.).sum() == mask.size, mask.unique()

            image = normalize_numpy(image)
            image = np.expand_dims(image, axis=-1)

            batch_index = i // self.slices
            sequence_index = i % self.slices

            batch_x[batch_index, sequence_index, ...] = image
            batch_y[batch_index, sequence_index, ...] = mask

        return batch_x, batch_y

    def on_epoch_end(self):
        pass

    def adjust_indices(self, batch_x, batch_y, batch_patients):
        dimension = len(batch_x)
        iterations = dimension // self.batch_size + 1

        new_batch_x = batch_x.copy()
        new_batch_y = batch_y.copy()

        for i in range(iterations):
            if i >= dimension:
                break
            patient = batch_patients[i]
            for j in range(self.slices):
                if (i * self.slices) + j >= dimension:
                    break
                if batch_patients[(i * self.slices) + j] != patient:
                    new_batch_x[(i * self.slices) + j] = new_batch_x[(i * self.slices) + j - 1]
                    new_batch_y[(i * self.slices) + j] = new_batch_y[(i * self.slices) + j - 1]

        return new_batch_x, new_batch_y
    def check_indices(self, indices):
        patients = [self.patients[i] for i in indices]
        print(indices)
        patients = np.array(patients)
        patients = patients.reshape((self.batch_size, self.slices))
        new_indices = indices.copy()
        # transform into numpy array
        new_indices = np.array(new_indices)
        new_indices = new_indices.reshape((self.batch_size, self.slices))

        for i in range(self.batch_size):
            patient = patients[i, 0]
            for j in range(self.slices):
                if patients[i, j] != patient:
                    new_indices[i, j] = new_indices[i, j-1]

        new_indices = new_indices.reshape((self.batch_size * self.slices)).tolist()
        return new_indices

