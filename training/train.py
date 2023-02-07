import os

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from data.input_data import ImageSegmentationDataset
from model.models import UNet

if __name__ == '__main__':
    batch_size = 4
    num_epochs = 10
    lr = 0.01
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    training_path = '/home/calde/Desktop/master-thesis-corino/code/data/csv_files/train.csv'
    validation_path = '/home/calde/Desktop/master-thesis-corino/code/data/csv_files/val.csv'
    training_dataset = ImageSegmentationDataset(csv_file=training_path, transform=transform)
    validation_dataset = ImageSegmentationDataset(csv_file=validation_path, transform=transform)
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    #%%

    model = UNet(num_classes=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print('starting')
    print(device)

    # add tqdm to show progress

    for epoch in tqdm(range(num_epochs)):
        for i, data in enumerate(train_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"epoch{epoch}")

        # Evaluate the model on the validation set
        with torch.no_grad():
            correct = 0
            total = 0
            for i, data in enumerate(valid_loader):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print("Epoch {}/{}, Accuracy: {:.2f}%".format(epoch + 1, num_epochs, accuracy))

    # Save the trained model
    torch.save(model.state_dict(), "model.pt")

