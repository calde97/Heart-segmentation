import os

# Change with your current HOME directory
HOME = '/mnt/datafast'
SOURCE_FOLDER = 'data-tesi-corino'
DATA_HOME = os.path.join(HOME, SOURCE_FOLDER)
IPERTROFIA_PATH = 'Data_ipertrofia'
AMILIODOSI_PATH = 'Data_amiloidosi_new'
STENOSI_PATH = 'Data_stenosi_new'

ALL_PATHS = [IPERTROFIA_PATH, AMILIODOSI_PATH, STENOSI_PATH]
DICOM = 'DICOM'
ROI = 'ROI'
CT = 'CT'
ROI_T = 'ROI_T.nrrd'

reading_order = {IPERTROFIA_PATH: False,
                 AMILIODOSI_PATH: True,
                 STENOSI_PATH: False}


MODE_AUGMENTATION_TRAIN = 'train'
MODE_AUGMENTATION_VAL = 'val'
MODE_AUGMENTATION_TRAIN_ENCODER = 'train_encoder'
DICE_LOSS = 'dice_loss'
BCE_LOSS = 'bce_loss'
UNET_MODEL = 'unet'
AUTOENCODER_MODEL = 'autoencoder'
UNET_MODEL_GENERIC = 'unet_generic'
UNET_LSTM = 'unet_lstm'