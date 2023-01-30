#%%
import imageio
from natsort import natsorted
import os
import nrrd
import matplotlib.pyplot as plt
import pydicom
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import constants

def create_mask_and_picture(folder:str, list_images, list_masks, names):
    # create folder if not exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    for index in range(len(list_masks)):
        plt.imshow(list_images[index], cmap=plt.cm.bone)
        plt.title(f'{names[index]}')
        plt.imshow(list_masks[index], alpha=0.25, cmap='gray')
        plt.savefig(f'{folder}/{index}.png')
        plt.clf()


def create_video_from_folder(src_folder:str, dest_folder:str, fps:int=1, name='video.mp4'):
    list_images = os.listdir(src_folder)
    list_images = natsorted(list_images)
    images = [plt.imread(f'{src_folder}/{s}') for s in list_images]
    imageio.mimsave(f'{dest_folder}/{name}', images, fps=fps)

    return list_images

def create_images_from_list(list_images, names,  dest_folder:str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for index, image in enumerate(list_images):
        plt.imshow(image, cmap=plt.cm.bone)
        plt.title(f'{names[index]}')
        plt.savefig(f'{dest_folder}/{index}.png')
        plt.clf()

def read_dicom(path:str, reverse_order=False):
    #read all files in the folder path_dicom
    dicom_file_names = os.listdir(path)
    #sort the files in the folder
    #dicom_file_names = natsorted(dicom_file_names)
    # append path dicom as prefix to the list_files_dicom
    complete_path_dicoms = [os.path.join(path, s) for s in dicom_file_names]
    #read all files with dcmread and put in a list
    data_dicom_list = [pydicom.dcmread(file) for file in complete_path_dicoms]
    data_dicom_list = sorted(data_dicom_list, key=lambda s: s.SliceLocation, reverse=reverse_order)
    list_pixel_array_images = [s.pixel_array for s in data_dicom_list]
    names = [s.filename for s in data_dicom_list]
    dict_images = {el.filename:el.pixel_array for el in data_dicom_list}
    # create dicti from names and list_pixel_array_images
    #dict_images = dict(zip(names, list_pixel_array_images))
    return names, list_pixel_array_images

def read_nrrd(path:str):
    data, header = nrrd.read(path)
    list_masks = [data[:, :, i].T for i in range(data.shape[2])]
    return list_masks

def get_dicom_order(path:str):
    list_path = path.split('/')
    if constants.AMILIODOSI_PATH in list_path:
        return True
    return False

#%%
#False
DATA_PATH = '../data-tesi-corino'
path_folder_dicom = os.path.join(DATA_PATH, 'Data_ipertrofia/DICOM/HCM13/CT')
file_nrrd = os.path.join(DATA_PATH, 'Data_ipertrofia/ROI/HCM13/ROI_T.nrrd')

#True
path_folder_dicom = os.path.join(DATA_PATH, 'Data_amiloidosi_new/DICOM/AM24/CT')
file_nrrd = os.path.join(DATA_PATH, 'Data_amiloidosi_new/ROI/AM24/ROI_T.nrrd')

'''
#False
path_folder_dicom = os.path.join(DATA_PATH, 'Data_stenosi_new/DICOM/AS052/CT')
file_nrrd = os.path.join(DATA_PATH, 'Data_stenosi_new/ROI/AS052/ROI_T.nrrd')'''


masks = read_nrrd(file_nrrd)
order = get_dicom_order(path_folder_dicom)
names, images = read_dicom(path_folder_dicom, order)
#create_images_from_list(images, names, 'all_hcm')
#%%
create_mask_and_picture('amiliososi', images, masks, names)
#%%
