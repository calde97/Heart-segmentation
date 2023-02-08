import os

HOME = '/home/calde/Desktop/master-thesis-corino'
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
