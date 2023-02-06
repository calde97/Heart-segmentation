import os

import imageio
import matplotlib.pyplot as plt
import nrrd
import pydicom
from natsort import natsorted

import constants


def create_mask_and_picture(folder: str, list_images, list_masks, names=None):
    # create folder if not exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    if names is None:
        names = [f'{index}' for index in range(len(list_images))]

    for index in range(len(list_masks)):
        plt.imshow(list_images[index], cmap=plt.cm.bone)
        plt.title(f'{names[index]}')
        plt.imshow(list_masks[index], alpha=0.25, cmap='gray')
        plt.savefig(f'{folder}/{index}.png')
        plt.clf()


def create_comparison_with_without_preprocessing(folder: str, list_images, list_images_without_preprocessing):
    # create folder if not exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    for index in range(len(list_images)):
        # plot the images side by side with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(list_images_without_preprocessing[index], cmap='gray')
        ax2.imshow(list_images[index], cmap='gray')
        plt.savefig(f'{folder}/{index}.png')
        plt.clf()


def create_video_from_folder(src_folder: str, dest_folder: str, fps: int = 1, name='video.mp4'):
    list_images = os.listdir(src_folder)
    list_images = natsorted(list_images)
    images = [plt.imread(f'{src_folder}/{s}') for s in list_images]
    imageio.mimsave(f'{dest_folder}/{name}', images, fps=fps)

    return list_images


def create_images_from_list(list_images, names, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for index, image in enumerate(list_images):
        plt.imshow(image, cmap=plt.cm.bone)
        plt.title(f'{names[index]}')
        plt.savefig(f'{dest_folder}/{index}.png')
        plt.clf()


def read_dicom(path: str, reverse_order=False, hu_transformation_flag=False, windowing_flag=False,
               window_level=40, window_width=400):
    # read all files in the folder path_dicom
    dicom_file_names = os.listdir(path)
    # sort the files in the folder
    # dicom_file_names = natsorted(dicom_file_names)
    # append path dicom as prefix to the list_files_dicom
    complete_path_dicoms = [os.path.join(path, s) for s in dicom_file_names]
    # read all files with dcmread and put in a list
    data_dicom_list = [pydicom.dcmread(file) for file in complete_path_dicoms]
    data_dicom_list = sorted(data_dicom_list, key=lambda s: s.SliceLocation, reverse=reverse_order)
    list_pixel_array_images = [s.pixel_array for s in data_dicom_list]

    if hu_transformation_flag:
        list_pixel_array_images = [transform_hu(image, s.RescaleSlope, s.RescaleIntercept)
                                   for image, s in zip(list_pixel_array_images, data_dicom_list)]

    if windowing_flag:
        list_pixel_array_images = [tranform_windowing(image, window_level, window_width)
                                   for image in list_pixel_array_images]

    file_names = [s.filename for s in data_dicom_list]

    return file_names, list_pixel_array_images


def read_single_dicom(path: str, hu_transformation_flag=True, windowing_flag=True,
                      window_level=40, window_width=400):
    dicom_file = pydicom.dcmread(path)
    image = dicom_file.pixel_array

    if hu_transformation_flag:
        image = transform_hu(image, dicom_file.RescaleSlope, dicom_file.RescaleIntercept)

    if windowing_flag:
        image = tranform_windowing(image, window_level, window_width)
    return image


def transform_hu(image, rescale_slope, rescale_intercept):
    return image * rescale_slope + rescale_intercept


def tranform_windowing(image, window_level, window_width):
    min_value = window_level - window_width // 2
    max_value = window_level + window_width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image


def read_nrrd(path: str):
    data, header = nrrd.read(path)
    list_masks = [data[:, :, i].T for i in range(data.shape[2])]
    return list_masks


def get_dicom_order(path: str):
    list_path = path.split('/')
    if constants.AMILIODOSI_PATH in list_path:
        return True
    return False
