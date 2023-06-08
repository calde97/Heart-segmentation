import os
import matplotlib.pyplot as plt
import nrrd
import torch
import torch.nn as nn
import torchmetrics
from natsort import natsorted
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import constants
from data.create_csv_data import get_category, get_patient
from data.custom_augmentations import get_val_transform
from data.input_data import ImageSegmentationDataset
from data.preprocessing import read_nrrd, get_dicom_order, read_dicom
from model.models import UNet, Autoencoder
from torchmetrics import Dice, Precision, Recall, F1Score
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import nrrd

def load_model(model_path):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = UNet(num_classes=1)

    model.load_state_dict(
        torch.load(model_path,
                   map_location=torch.device(device)))

    model.to(device)
    model.eval()
    return model, device



PATH_DATA = '/home/calde/Desktop/master-thesis-corino/data-tesi-corino'
patient = 'AM08'
# category = 'Data_stenosi_new'
# category = 'Data_ipertrofia'
category = 'Data_amiloidosi_new'



def run_inference(model, path, device, progress_bar, destination_path='none', root=None):
    #path = f'/home/calde/Desktop/master-thesis-corino/data-tesi-corino/{category}/DICOM/{patient}/CT'
    # divide path in '/' tokens and get the category as the -4th element
    category = path.split('/')[-4]
    # get the patient as the -2nd element
    patient = path.split('/')[-2]


    order = get_dicom_order(path)
    names, images = read_dicom(path, order, hu_transformation_flag=True, windowing_flag=True)
    transforms = A.Compose([
        A.Resize(256, 256),
        ToTensorV2()
    ])

    # apply the transforms to the images
    images = [transforms(image=image)['image'] for image in images]

    # add the batch dimension to the images
    images = [image.unsqueeze(0) for image in images]
    outputs = []


    def normalize(image):
        # get max of image
        max_value = torch.max(image)
        # get min of image
        min_value = torch.min(image)
        # normalize image
        image = (image - min_value) / (max_value - min_value)
        return image


    for i in tqdm(range(len(images))):
        images[i] = normalize(images[i])
        images[i] = images[i].float()

        # perform inference on the images
        with torch.no_grad():
            images[i] = images[i].to(device)
            out = model(images[i])
            out = torch.sigmoid(out)
            outputs.append(out)

        print(i)
        print('aosnfoasfnoasfn')
        progress_bar['value'] = (i + 1) / len(images) * 100  # Update the progress bar value
        root.update()
        print(progress_bar['value'])
        #root.update()
    # transform outputs to numpy
    outputs = [output.cpu().numpy() for output in outputs]

    # concatenate the outputs
    outputs = np.concatenate(outputs, axis=0)

    # remove the batch dimension
    outputs = outputs.squeeze(1)


    images = [image.cpu().numpy() for image in images]
    images = np.concatenate(images, axis=0)
    images = images.squeeze(1)

    if destination_path == 'none':
        inference_path = 'inference_test'
        dst_path = os.path.join(constants.DATA_HOME, inference_path)
    else:
        dst_path = destination_path


    if not os.path.exists(os.path.join(dst_path, patient)):
        os.makedirs(os.path.join(dst_path, patient))

    nrrd.write(f'{dst_path}/{patient}/image.nrrd', images)
    nrrd.write(f'{dst_path}/{patient}/prediction.nrrd', outputs)

    return patient

