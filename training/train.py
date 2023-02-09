from datetime import datetime
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from data.input_data import ImageSegmentationDataset
from model.models import UNet


def IoU(predicted, labels):
    true_positives = (predicted == 1) & (labels == 1)
    false_positives = (predicted == 1) & (labels == 0)
    false_negatives = (predicted == 0) & (labels == 1)
    IoU = true_positives.sum() / (true_positives.sum() + false_positives.sum() + false_negatives.sum())
    return IoU


def dice_loss(predicted, labels):
    smooth = 1.
    true_positives = (predicted == 1) & (labels == 1)
    false_positives = (predicted == 1) & (labels == 0)
    false_negatives = (predicted == 0) & (labels == 1)

    true_positives = true_positives.float()
    false_positives = false_positives.float()
    false_negatives = false_negatives.float()

    dice_coefficient = (2. * true_positives.sum() + smooth) / (2. * true_positives.sum() + false_positives.sum() +
                                                               false_negatives.sum() + smooth)
    return 1 - dice_coefficient


if __name__ == '__main__':
    batch_size = 1
    num_epochs = 100
    lr = 0.0001
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=180)
    ])
    training_path = '../data/csv_files/train.csv'
    validation_path = '../data/csv_files/val.csv'

    limit = None

    training_dataset = ImageSegmentationDataset(csv_file=training_path, transform=transform, limit_for_testing=limit,
                                                apply_hu_transformation=True, apply_windowing=True)
    validation_dataset = ImageSegmentationDataset(csv_file=training_path, transform=transform, limit_for_testing=limit)
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(num_classes=1).to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameters: ", total_params)
    # add tqdm to show progress

    best_val_loss = 1000
    best_val_iou = 0

    for epoch in (outer_bar := tqdm(range(num_epochs))):
        total_training_loss = 0
        total_training_iou = 0
        for i, data in (inner_bar := tqdm(enumerate(train_loader))):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # apply sigmoid to the output
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, labels)
            # get the prediction by comparing the output with 0.5
            predicted = (outputs > 0.5).float()
            iou = IoU(predicted, labels)
            loss.backward()
            total_training_loss += loss.item()
            total_training_iou += iou
            optimizer.step()

            inner_bar.set_description(f"Loss: {loss.item():.2f}, IoU: {iou:.2f}")

        mean_loss = total_training_loss / len(train_loader)
        outer_bar.set_description(f"Epoch: {epoch + 1}/{num_epochs}, Training loss: {mean_loss:.2f}")
        outer_bar.set_description(f"Epoch: {epoch + 1}/{num_epochs}, Training IoU: {total_training_iou / len(train_loader):.2f}")



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
                loss = criterion(outputs, labels)
                val_loss_total += loss.item()
                predicted = (outputs > 0.5).float()
                iou += IoU(predicted, labels)
                index += 1
            mean_iou = iou / len(valid_loader)
            mean_val_loss = val_loss_total / len(valid_loader)
            print(f"Validation loss: {mean_val_loss:.2f}, Validation IoU: {mean_iou:.2f}")
            print(f'Mean IoU: {mean_iou:.2f}')

        if mean_val_loss < best_val_loss and mean_val_loss < 0.80:
            best_val_loss = mean_val_loss
            torch.save(model.state_dict(), f"model{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.pth")
            print("Model saved")

