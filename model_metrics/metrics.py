import torch
import torch.nn as nn
import torchmetrics
from data import constants
def dice_loss(y_pred, y_true):
    smooth = 1.
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice_score
    return dice_loss




def IoU():
    iou = torchmetrics.JaccardIndex(num_classes=1, task='binary')
    return iou


def get_criterion_from_name(criterion_name):
    if criterion_name == constants.DICE_LOSS:
        return dice_loss
    elif criterion_name == constants.BCE_LOSS:
        return nn.BCELoss()
    else:
        raise ValueError('Criterion not found')
