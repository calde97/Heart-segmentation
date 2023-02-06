import pandas as pd
import numpy as np
import os
'''
Divide the dataset in train, validation and test. based on the patient.
The patient is the same for all the images of the same patient so that we can use it to divide the dataset
'''

if __name__ == '__main__':
    df = pd.read_csv('csv_files/whole_dataset.csv')
    # get the unique patients
    patients = df['patient'].unique()
    # shuffle the patients
    np.random.shuffle(patients)
    # get the number of patients
    n_patients = len(patients)
    # get the number of patients for each set
    n_train = int(n_patients * 0.75)
    n_val = int(n_patients * 0.15)
    n_test = n_patients - n_train - n_val
    # get the patients for each set
    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]
    # create the sets
    train_df = df[df['patient'].isin(train_patients)]
    val_df = df[df['patient'].isin(val_patients)]
    test_df = df[df['patient'].isin(test_patients)]
    # get the percentage of each set
    print('train: ', len(train_df) / len(df))
    print('val: ', len(val_df) / len(df))
    print('test: ', len(test_df) / len(df))
    # add column X as prefix_path + filename
    train_df['X_path'] = train_df['prefix_path'] + os.sep + train_df['filename']
    val_df['X_path'] = val_df['prefix_path'] + os.sep + val_df['filename']
    test_df['X_path'] = test_df['prefix_path'] + os.sep + test_df['filename']
    # add column y as mask_path + serial_number
    train_df['y_path'] = train_df['mask_path'].apply(lambda x: os.sep.join(x.split(os.sep)[:-1])) + os.sep + train_df[
        'serial_number'].apply(lambda x: str(x).zfill(3)) + '.nrrd'
    val_df['y_path'] = val_df['mask_path'].apply(lambda x: os.sep.join(x.split(os.sep)[:-1])) + os.sep + val_df[
        'serial_number'].apply(lambda x: str(x).zfill(3)) + '.nrrd'
    test_df['y_path'] = test_df['mask_path'].apply(lambda x: os.sep.join(x.split(os.sep)[:-1])) + os.sep + test_df[
        'serial_number'].apply(lambda x: str(x).zfill(3)) + '.nrrd'
    # delete the mask_path and prefix_path columns
    train_df.drop(['mask_path', 'prefix_path'], axis=1, inplace=True)
    val_df.drop(['mask_path', 'prefix_path'], axis=1, inplace=True)
    test_df.drop(['mask_path', 'prefix_path'], axis=1, inplace=True)
    # put X_path and y_path as first and second columns
    train_df = train_df[['X_path', 'y_path', 'category', 'patient', 'filename', 'serial_number', 'id_name']]
    val_df = val_df[['X_path', 'y_path', 'category', 'patient', 'filename', 'serial_number', 'id_name']]
    test_df = test_df[['X_path', 'y_path', 'category', 'patient', 'filename', 'serial_number', 'id_name']]
    # save the sets
    train_df.to_csv('csv_files/trainmm.csv', index=False)
    val_df.to_csv('csv_files/valmm.csv', index=False)
    test_df.to_csv('csv_files/testmm.csv', index=False)
