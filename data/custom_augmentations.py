import albumentations as A
from albumentations.pytorch import ToTensorV2
from data import constants

'''
Custom augmentation for the deep learning training.
'''


def get_train_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.SafeRotate(limit=90, p=0.2, border_mode=0, value=0, mask_value=0),
        ToTensorV2(),
    ])


def get_train_autoencoder_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.GaussNoise(var_limit=(0.0, 0.5), mean=0, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=5, p=1, border_mode=0, value=0, mask_value=0),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(256, 256),
        ToTensorV2(),
    ])


def get_augmentation_transform(mode):
    if mode == constants.MODE_AUGMENTATION_TRAIN:
        return get_train_transform()
    if mode == constants.MODE_AUGMENTATION_VAL:
        return get_val_transform()
    if mode == constants.MODE_AUGMENTATION_TRAIN_ENCODER:
        return get_train_autoencoder_transform()
    raise NotImplementedError
