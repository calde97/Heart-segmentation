import nrrd
import numpy as np
import os
import pandas as pd
from natsort import natsorted
from tqdm import tqdm

from data import constants
from data.preprocessing import read_nrrd, get_dicom_order, read_dicom

'''
Create a csv file from the DICOM and ROI folders. A csv file with all the slices of the patient is created
'''


def get_category(path):
    tokens = path.split(os.sep)
    return tokens[-4]


def get_patient(path):
    tokens = path.split(os.sep)
    return tokens[-2]


if __name__ == '__main__':
    dicom_category_paths = [os.path.join(constants.DATA_HOME, path, constants.DICOM) for path in constants.ALL_PATHS]
    patients_per_category = []
    for category in dicom_category_paths:
        patients = [os.path.join(category, patient, constants.CT) for patient in natsorted(os.listdir(category))]
        patients_per_category.extend(patients)

    masks = [os.path.join(constants.DATA_HOME, path, constants.ROI) for path in constants.ALL_PATHS]
    mask_patients_per_category = []
    for category in masks:
        patients = [os.path.join(category, patient, constants.ROI_T) for patient in natsorted(os.listdir(category))]
        mask_patients_per_category.extend(patients)

    data = {'image_path': patients_per_category, 'mask_path': mask_patients_per_category}
    df = pd.DataFrame(data)

    data = {'prefix_path': [], 'filename': [], 'serial_number': [], 'mask_path': []}

    # use tqdm to create a loading bar
    for name_path, mask_path in tqdm(df[['image_path', 'mask_path']].values):
        masks = read_nrrd(mask_path)
        order = get_dicom_order(name_path)
        names, images = read_dicom(name_path, order, hu_transformation_flag=True, windowing_flag=True)
        index = 0
        # get only the images and mask that have a mask ( if the mask is empty it is not used )
        for i, mask in enumerate(masks):
            if np.sum(mask) > 0:
                data['prefix_path'].append(name_path)
                data['filename'].append(names[i].split(os.sep)[-1])
                data['serial_number'].append(index)
                data['mask_path'].append(mask_path)
                # write the mask in nrrd format.
                mask_path_without_filename = mask_path.split(os.sep)[:-1]
                mask_path_without_filename = os.sep.join(mask_path_without_filename)
                mask_file_name = os.path.join(mask_path_without_filename, f'{str(index).zfill(3)}.nrrd')
                nrrd.write(mask_file_name, mask)
                index += 1

    df = pd.DataFrame(data)

    # add column category, patient and id_name to the dataframe
    df['category'] = df['prefix_path'].apply(get_category)
    df['patient'] = df['prefix_path'].apply(get_patient)
    df['id_name'] = df['category'] + '__' + df['patient'] + '__' + df['filename'] + '__' + df['serial_number'].astype(
        str)

    df.to_csv('csv_files/whole_data.csv', index=False)
