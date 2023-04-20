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
# %%

cwd = os.getcwd()
if not cwd.endswith('Heart-segmentation'):
    os.chdir('..')

print(os.getcwd())

# Load the datasets
if os.path.exists('data/csv_files/train.csv'):
    train_path = 'data/csv_files/train.csv'
    val_path = 'data/csv_files/val-fix.csv'
else:
    train_path = '../data/csv_files/train.csv'
    val_path = '../data/csv_files/val-fix.csv'

transform = get_val_transform()
val_dataset = ImageSegmentationDataset(csv_file=val_path,
                                       transform=transform,
                                       limit_for_testing=73)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# %%

# Load the model

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = UNet(num_classes=1)

try:
    model.load_state_dict(
        torch.load('training/model_unet_yaml_0.76077115535736082023_02_19__05_35_45.pth',
                   map_location=torch.device(device)))
except:
    model.load_state_dict(
        torch.load('/mnt/data/model_unet_yaml_0.76703214645385742023_02_20__02_06_37.pth',
                   map_location=torch.device(device)))

#lstm path = model_unet_yaml_extended0.14326076843799692023_02_24__10_13_30.pth
# unet 2.5d = model_unet_yaml_0.76703214645385742023_02_20__02_06_37.pth
# baseline unet = model_unet_yaml_0.76077115535736082023_02_19__05_35_45.pth

model.to(device)
model.eval()

# %%

dicom_category_paths = [os.path.join(constants.DATA_HOME, path, constants.DICOM) for path in constants.ALL_PATHS]

# %%
patients_per_category = []
for category in dicom_category_paths:
    patients = [os.path.join(category, patient, constants.CT) for patient in natsorted(os.listdir(category))]
    patients_per_category.extend(patients)
# %%
masks = [os.path.join(constants.DATA_HOME, path, constants.ROI) for path in constants.ALL_PATHS]
mask_patients_per_category = []
for category in masks:
    patients = [os.path.join(category, patient, constants.ROI_T) for patient in natsorted(os.listdir(category))]
    mask_patients_per_category.extend(patients)

# %%
data = {'image_path': patients_per_category}
df = pd.DataFrame(data)

# %%
patient = 'AM10'
#category = 'Data_ipertrofia'
#category = 'Data_amiloidosi_new'
category = 'Data_stenosi_new'
path = f'/mnt/datafast/data-tesi-corino/{category}/DICOM/{patient}/CT'
mask_path = f'/mnt/datafast/data-tesi-corino/{category}/ROI/{patient}/ROI_T.nrrd'
order = get_dicom_order(path)
names, images = read_dicom(path, order, hu_transformation_flag=True, windowing_flag=True)
masks = read_nrrd(mask_path)


# %%
import albumentations as A
from albumentations.pytorch import ToTensorV2

transforms = A.Compose([
    A.Resize(256, 256),
    ToTensorV2()
])


# apply the transforms to the images
images = [transforms(image=image)['image'] for image in images]

# add the batch dimension to the images
images = [image.unsqueeze(0) for image in images]

gt_masks = [transforms(image=mask)['image'] for mask in masks]
gt_masks = [mask.unsqueeze(0) for mask in gt_masks]


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



# transform outputs to numpy
outputs = [output.cpu().numpy() for output in outputs]

# concatenate the outputs
outputs = np.concatenate(outputs, axis=0)


# remove the batch dimension
outputs = outputs.squeeze(1)



# save the outputs in a numpy file for later use
#np.save('/mnt/data/outputss30.npy', outputs)

#%%
# transform the images to numpy. We need to remove the batch dimension. concatenate the images
images = [image.cpu().numpy() for image in images]
images = np.concatenate(images, axis=0)
images = images.squeeze(1)

#%%

gt_masks = [mask.cpu().numpy() for mask in gt_masks]
gt_masks = np.concatenate(gt_masks, axis=0)
gt_masks = gt_masks.squeeze(1)

#%%
import nrrd
nrrd.write(f'/mnt/data/ct_scan_{patient}.nrrd', images)
nrrd.write(f'/mnt/data/bm_mask_{patient}.nrrd', outputs)
nrrd.write(f'/mnt/data/gt_mask_{patient}.nrrd', gt_masks)
#nrrd.write(f'/mnt/data/lstm_mask_{patient}.nrrd', outputs)
#%%
nrrd.write(f'/mnt/data/gt_mask_{patient}.nrrd', gt_masks)

#%%
# save the images in a numpy file for later use
#np.save('/mnt/data/imagess30.npy', images)

# %%
i = 1000
for image, out, name in zip(images, outputs, names):
    out = out.cpu().numpy()
    out = out.squeeze()
    plt.imshow(image[0, 0, :, :].cpu().numpy(), cmap='gray')
    plt.imshow(out, cmap='gray', alpha=0.3)
    # remove everything after . in the name
    name = name.split('.')[0]
    name = name.split('/')[-1]
    print(name)
    plt.savefig(f'/mnt/data/keras_pictures/HCM05/{i}.png')
    i += 1



# %%
plt.imshow(inputtino[0, 0, :, :].cpu().numpy(), cmap='gray')
plt.show()
# %%

outputs = torch.sigmoid(outputs)
outputs = (outputs > 0.5).float()

plt.imshow(outputs[0, 0, :, :].cpu().numpy())
plt.show()

# %%
print(3)
# %%
validation_path = '/home/calderon/tesi/Heart-segmentation/data/csv_files/val.csv'
val_dataset = ImageSegmentationDataset(validation_path, transform=transforms, limit_for_testing=5)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# %%

with torch.no_grad():
    for i, (images, masks, pat, sn) in enumerate(val_loader):
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        outputs = (outputs > 0.5).float()
        print(outputs.shape)
        print(masks.shape)
        break

# %%

data = {'prefix_path': [], 'filename': [], 'serial_number': []}

# use tqdm to create a loading bar
for name_path in tqdm(df[['image_path']].values):
    order = get_dicom_order(name_path)
    names, images = read_dicom(name_path, order, hu_transformation_flag=True, windowing_flag=True)
    index = 0
    break

    # %%
    # get only the images and mask that have a mask ( if the mask is empty it is not used )
'''    for i, mask in enumerate(masks):
        data['prefix_path'].append(name_path)
        data['filename'].append(names[i].split(os.sep)[-1])
        data['serial_number'].append(index)
        data['mask_path'].append(mask_path)
        # write the mask in nrrd format.
        mask_path_without_filename = mask_path.split(os.sep)[:-1]
        mask_path_without_filename = os.sep.join(mask_path_without_filename)
        mask_file_name = os.path.join(mask_path_without_filename, f'{str(index).zfill(3)}.nrrd')
        nrrd.write(mask_file_name, mask)
        index += 1'''

df = pd.DataFrame(data)

# add column category, patient and id_name to the dataframe
df['category'] = df['prefix_path'].apply(get_category)
df['patient'] = df['prefix_path'].apply(get_patient)
df['id_name'] = df['category'] + '__' + df['patient'] + '__' + df['filename'] + '__' + df['serial_number'].astype(
    str)
