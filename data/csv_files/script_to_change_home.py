import pandas as pd

import os
from data import constants
import argparse
'''
Change the path for the HOME if we use csv files produced on another machine.
'''

def change_home_path(path, new_home_path):
    # split the path by os.sep
    path = path.split(os.sep)
    postfix = os.path.join(*path[5:])
    new_path = os.path.join(new_home_path, postfix)

    return new_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('home', type=str, default=constants.HOME, help='The new home path')



    df_train = pd.read_csv('data/csv_files/train.csv')
    df_test = pd.read_csv('data/csv_files/test.csv')
    df_val = pd.read_csv('data/csv_files/val.csv')

    # change_home_path to X_path and y_path
    df_train['X_path'] = df_train['X_path'].apply(lambda x: change_home_path(x, constants.HOME))
    df_train['y_path'] = df_train['y_path'].apply(lambda x: change_home_path(x, constants.HOME))

    df_test['X_path'] = df_test['X_path'].apply(lambda x: change_home_path(x, constants.HOME))
    df_test['y_path'] = df_test['y_path'].apply(lambda x: change_home_path(x, constants.HOME))

    df_val['X_path'] = df_val['X_path'].apply(lambda x: change_home_path(x, constants.HOME))
    df_val['y_path'] = df_val['y_path'].apply(lambda x: change_home_path(x, constants.HOME))

    #writing the new csv files
    df_train.to_csv('data/csv_files/train.csv', index=False)
    df_test.to_csv('data/csv_files/test.csv', index=False)
    df_val.to_csv('data/csv_files/val.csv', index=False)