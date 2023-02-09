import torch
import numpy as np
from torch import nn

from model.models import UNet

model = UNet(num_classes=1)
# load model..pt state dict
model.load_state_dict(torch.load('/home/calde/Desktop/master-thesis-corino/code/model_binaries/model.pt',
                                 map_location=torch.device('cpu')))
print(model)

# load test data from csv
import pandas as pd

# evaluate the model on the test data
from data.preprocessing import read_single_dicom
import nrrd
from data.input_data import ImageSegmentationDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor(),
])

validation_path = '/home/calde/Desktop/master-thesis-corino/code/data/csv_files/val.csv'
val_dataset = ImageSegmentationDataset(validation_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

meanIoU = 0
max_iteration = 1

model = UNet(num_classes=1)
# load model
model.load_state_dict(
    torch.load('/home/calde/Desktop/master-thesis-corino/code/model/modello.pth', map_location=torch.device('cpu')))

# %%
criterion = nn.BCELoss()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model.eval()


def IoU(predicted, labels):
    true_positives = (predicted == 1) & (labels == 1)
    false_positives = (predicted == 1) & (labels == 0)
    false_negatives = (predicted == 0) & (labels == 1)
    IoU = true_positives.sum() / (true_positives.sum() + false_positives.sum() + false_negatives.sum())
    return IoU


total_loss = 0
meanIoU = 0

with torch.no_grad():
    for (images, labels) in (bar := (tqdm(val_loader))):
        outputs = model(images)
        # pass sigmoid to get probabilities
        outputs = torch.sigmoid(outputs)
        predicted = (outputs > 0.5).float()
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        iou = IoU(predicted, labels)
        meanIoU += iou

        bar.set_description(f"loss: {loss.item():.4f}, IoU: {iou:.4f}")

    print("meanIoU: ", meanIoU / len(val_loader))
    print("loss: ", loss / len(val_loader))
