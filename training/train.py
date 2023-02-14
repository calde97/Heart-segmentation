import os
from datetime import datetime

import torch
import wandb
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from data.input_data import ImageSegmentationDataset
from model.models import UNet
import sys
import traceback
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchmetrics


# %%


'''def IoU(predicted, labels):
    true_positives = (predicted == 1) & (labels == 1)
    false_positives = (predicted == 1) & (labels == 0)
    false_negatives = (predicted == 0) & (labels == 1)
    IoU = true_positives.sum() / (true_positives.sum() + false_positives.sum() + false_negatives.sum())
    return IoU'''


def dice_loss(y_pred, y_true):
    smooth = 1.
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice_score
    return dice_loss


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        batch_size = config.batch_size
        lr = config.learning_rate
        criterion = nn.BCELoss()
        model = UNet(num_classes=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        num_epochs = 100
        IoU = torchmetrics.JaccardIndex(num_classes=1, task='binary')

        transforms_train = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=5, p=1, border_mode=0, value=0, mask_value=0),
            ToTensorV2(),
        ])

        transforms_val = A.Compose([
            A.Resize(256, 256),
            ToTensorV2(),
        ])

        training_path = '../data/csv_files/train.csv'
        validation_path = '../data/csv_files/val.csv'

        flag_testing = False
        if flag_testing:
            shuffle_training = False
            shuffle_validation = False
            limit = 10

        else:
            print("Training on the whole dataset")
            shuffle_training = True
            shuffle_validation = False
            limit = None

        training_dataset = ImageSegmentationDataset(csv_file=training_path, transform=transforms_train,
                                                    limit_for_testing=limit,
                                                    apply_hu_transformation=True, apply_windowing=True)
        validation_dataset = ImageSegmentationDataset(csv_file=validation_path, transform=transforms_val,
                                                      limit_for_testing=limit)

        if flag_testing:
            validation_dataset = training_dataset

        train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle_training)
        valid_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle_validation)


        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of learnable parameters: ", total_params)

        best_val_loss = 1000
        best_iou = 0

        number_of_epochs_without_improvement = 0


        for epoch in (outer_bar := tqdm(range(num_epochs))):
            total_training_loss = 0
            total_training_iou = 0
            for i, data in (enumerate(train_loader)):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                # apply sigmoid to the output
                outputs = torch.sigmoid(outputs)
                torch.where(outputs > 0.5, torch.round(outputs), outputs)
                loss = dice_loss(outputs, labels)
                # get the prediction by comparing the output with 0.5
                #predicted = (outputs > 0.5).float()
                iou = IoU(outputs, labels)
                loss.backward()
                total_training_loss += loss.item()
                total_training_iou += iou
                optimizer.step()

                #inner_bar.set_description(f"Loss: {loss.item():.2f}, IoU: {iou:.2f}")

            mean_loss = total_training_loss / len(train_loader)
            outer_bar.set_description(f"Epoch: {epoch + 1}/{num_epochs}, Training loss: {mean_loss:.2f}")
            outer_bar.set_description(
                f"Epoch: {epoch + 1}/{num_epochs}, Training IoU: {total_training_iou / len(train_loader):.2f}")

            wandb.log({"Training loss": mean_loss, "Training IoU": total_training_iou / len(train_loader)})

            # Evaluate the model on the validation set
            with torch.no_grad():
                correct = 0
                total = 0
                index = 0
                iou = 0
                val_loss_total = 0
                for i, data in enumerate(valid_loader):
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    outputs = torch.sigmoid(outputs)
                    torch.where(outputs > 0.5, torch.round(outputs), outputs)
                    loss = dice_loss(outputs, labels)
                    val_loss_total += loss.item()
                    #predicted = (outputs > 0.5).float()
                    iou += IoU(outputs, labels)
                    index += 1
                mean_iou = iou / len(valid_loader)
                mean_val_loss = val_loss_total / len(valid_loader)
                validation_loss = mean_val_loss
                outer_bar.set_description(f"Validation loss: {mean_val_loss:.2f}, Validation IoU: {mean_iou:.2f}")
                print(f"Validation loss: {mean_val_loss:.2f}, Validation IoU: {mean_iou:.2f}")
                print(f'Mean IoU: {mean_iou:.2f}')

                wandb.log({"Validation loss": mean_val_loss, "Validation IoU": mean_iou})

            number_of_epochs_without_improvement += 1

            if mean_iou > best_iou:
                number_of_epochs_without_improvement = 0
                best_iou = mean_iou
                torch.save(model.state_dict(), f"model-2599-{mean_iou}{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.pth")
                print("Model saved")

            if number_of_epochs_without_improvement >= 30:
                print("Early stopping")
                break





def main():
    print(os.getcwd())
    wandb.login()
    # Load the YAML file into a dictionary
    with open("yaml_files/first.yaml", "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_config, project="Polimi-tesi-corino-albumentations-dice-loss")
    wandb.agent(sweep_id, train, count=5)



if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)
