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
    batch_size = 8
    num_epochs = 10
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

    training_dataset = ImageSegmentationDataset(csv_file=training_path, transform=transform, limit_for_testing=5,
                                                apply_hu_transformation=True, apply_windowing=True)
    validation_dataset = ImageSegmentationDataset(csv_file=training_path, transform=transform, limit_for_testing=5)
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    model = UNet(num_classes=1).to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameters: ", total_params)
    # add tqdm to show progress

    best_val_loss = np.inf
    best_val_iou = 0

    for epoch in tqdm(range(num_epochs)):
        for i, data in enumerate(train_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # apply sigmoid to the output
            outputs = torch.sigmoid(outputs)
            # get the prediction by comparing the output with 0.5
            predicted = (outputs > 0.5).float()
            loss = criterion(outputs, labels)
            iou = IoU(predicted, labels)
            loss.backward()
            optimizer.step()

            print(f'training loss: {loss}')
            print(f'training IoU: {iou}')

        # Evaluate the model on the validation set
        with torch.no_grad():
            correct = 0
            total = 0
            index = 0
            iou = 0
            for i, data in enumerate(valid_loader):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                predicted = (outputs > 0.5).float()
                loss = criterion(outputs, labels)
                total += (labels.size(-1) * labels.size(-2))
                correct += (predicted == labels).sum().item()
                iou += IoU(predicted, labels)
                index += 1
            accuracy = 100 * correct / total
            mean_iou = iou / index
            print("Epoch {}/{}, Accuracy: {:.2f}%".format(epoch + 1, num_epochs, accuracy))
            print("Epoch {}/{}, Mean IoU: {:.2f}%".format(epoch + 1, num_epochs, mean_iou))
            print("Epoch {}/{}, Validation loss: {:.2f}%".format(epoch + 1, num_epochs, loss))

            # get current date in python with format dd/mm/YY

            if loss < best_val_loss:
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
                best_val_loss = loss
                torch.save(model.state_dict(), f"{dt_string}-{epoch}model.pt")
                print("Model saved")
