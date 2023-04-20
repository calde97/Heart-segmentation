import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.custom_augmentations import get_val_transform
from data.input_data import ImageSegmentationDataset
from model.models import UNet, Autoencoder
from torchmetrics import Dice, Precision, Recall, F1Score
import numpy as np
import pandas as pd
#%%
def plot_prediction_and_gt(images, masks, predictions, metrics_iou, latent_loss=None, save_flag=False,
                           model_name='None', patient=None, serial_numbers=None):
    outputs_np = predictions.cpu().numpy()
    masks_np = masks.cpu().numpy()
    images_np = images.cpu().numpy()

    for index in range(images.shape[0]):
        # Create a single plot with two columns and one row
        fig, axs = plt.subplots(1, 3, figsize=(24, 8))

        # Plot the first image on the left
        axs[0].imshow(images_np[index, 0, :, :], cmap='gray')
        axs[0].set_title('Original Image')

        # Plot the second image on the right
        axs[1].imshow(images_np[index, 0, :, :], cmap='gray')
        axs[1].imshow(masks_np[index, 0, :, :], alpha=0.4, cmap='gray')
        axs[1].set_title('Ground truth')

        # Plot the third image on the right
        axs[2].imshow(images_np[index, 0, :, :], cmap='gray')
        axs[2].imshow(outputs_np[index, 0, :, :], alpha=0.4, cmap='gray', vmin=0.5)
        axs[2].set_title(f'Dice : {metrics_iou[index]:.4f} \nLatent loss :{latent_loss[index]:.4f}')

        # Set the plot width a little greater than the sum of the two subplots
        fig_width = axs[0].get_window_extent().width + axs[1].get_window_extent().width + \
                    axs[2].get_window_extent().width + 10
        fig.set_size_inches(fig_width / 80, 8)  # 80 pixels per inch (dpi)
        if save_flag:
            save_path = os.path.join('dash-visualizations', 'assets', 'models_evaluation', model_name, patient)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            name_file = os.path.join(save_path, f'{serial_numbers[index]}.png')
            plt.savefig(name_file)
        else:
            plt.show()



if __name__ == '__main__':

    cwd = os.getcwd()
    if not cwd.endswith('Heart-segmentation'):
        os.chdir('..')

    print(os.getcwd())

    # Load the datasets
    if os.path.exists('data/csv_files/train.csv'):
        train_path = 'data/csv_files/train.csv'
        val_path = 'data/csv_files/test.csv'
    else:
        train_path = '../data/csv_files/train.csv'
        val_path = '../data/csv_files/test.csv'

    transform = get_val_transform()
    val_dataset = ImageSegmentationDataset(csv_file=val_path,
                                           transform=transform)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Load the model

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model = UNet(num_classes=1)
    try:
        model.load_state_dict(
            torch.load('training/model_unet_yaml_0.76077115535736082023_02_19__05_35_45.pth',
                       map_location=torch.device(device)))
    except:
        model.load_state_dict(
            torch.load('../training/model_unet_yaml_0.76077115535736082023_02_19__05_35_45.pth',
                       map_location=torch.device(device)))

    model.to(device)
    model.eval()

    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(
        torch.load('training/model_0.83041048049926762023_02_18__12_51_37.pth', map_location=torch.device('cpu')))
    autoencoder.eval()
    encoder = autoencoder.encoder
    ddl = torch.nn.Sequential(encoder, *list(autoencoder.flatten.children())[:2])
    ddl.eval()

    # Do inference on the validation set
    iou = torchmetrics.JaccardIndex(num_classes=1, task='binary')
    iou_for_all = []
    dice_for_all = []
    precision_for_all = []
    recall_for_all = []
    f1_for_all = []
    latent_losses = []


    for images, masks, patient_names, serial_numbers in tqdm(val_loader):
        # transform sn into list of ints
        serial_numbers = [int(i) for i in serial_numbers]
        metrics_iou = []
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            outputs = nn.Sigmoid()(outputs)
            # loop through each output and mask and calculate the iou for each
            for i in range(outputs.shape[0]):
                iou(outputs[i, :, :, :], masks[i, :, :, :])
                metrics_iou.append(iou.compute())
                # transform masks from tensor of float into tensor of int by casting it
                masks_integer = torch.tensor(masks[i, :, :, :].cpu().numpy(), dtype=torch.int)
                dice_coefficient = Dice()(outputs[i, :, :, :], masks_integer)
                dice_for_all.append(dice_coefficient)
                precision_coeff = Precision(task='binary', num_classes=1, )(outputs[i, :, :, :], masks_integer)
                precision_for_all.append(precision_coeff)
                recall_coeff = Recall(task='binary', num_classes=1, )(outputs[i, :, :, :], masks_integer)
                recall_for_all.append(recall_coeff)
                # latent error
                ls_gt = ddl(masks)
                ls_pred = ddl(outputs)
                # sigmoid
                ls_gt = nn.Sigmoid()(ls_gt)
                ls_pred = nn.Sigmoid()(ls_pred)
                loss_bce = nn.BCELoss()(ls_pred, ls_gt)
                # get float value and append to list
                latent_losses.append(float(loss_bce))

                # transform metrics_iou to list of floats
            dices = [float(dice_coefficient)]
            latents = [float(loss_bce)]
            iou_for_all.extend(metrics_iou)

            try:
                plot_prediction_and_gt(images, masks, outputs, dices,
                                       save_flag=True, model_name='unet2-test', latent_loss=latents,
                                       patient=patient_names[0], serial_numbers=serial_numbers)
            except:
                print('error')
                continue



    iou_for_all = [float(i) for i in iou_for_all]
    dice_for_all = [float(i) for i in dice_for_all]
    precision_for_all = [float(i) for i in precision_for_all]
    recall_for_all = [float(i) for i in recall_for_all]



    # read evaluation metrics from file
    df = pd.read_csv('data/csv_files/test.csv')
    # append new metrics
    df['dice'] = dice_for_all
    df['precision'] = precision_for_all
    df['recall'] = recall_for_all
    df['iou'] = iou_for_all
    latent_losses = (latent_losses - np.min(latent_losses)) / (
                np.max(latent_losses) - np.min(latent_losses))

    df['latent'] = latent_losses

    # only leave the columns patient, serial_number, dice, precision, recall, iou
    df = df[['patient', 'serial_number', 'dice', 'precision', 'recall', 'iou', 'latent']]

    df.to_csv('evaluation/csv_files/unet2-test.csv', index=False)

    # save the iou_for_all into a numpy file for later use
    #np.save('iou_for_all2.npy', iou_for_all)