import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


class FEEDER():
    def __init__(self, config, feed_function, mode='train'):
        self.config = config
        self.feed_function = feed_function

        if self.config.DATASET['TRAIN_TEST_SPLIT']:
            self.split_dset()

    '''
		Splits The Dataset In Train And Valid Sets
	'''

    def split_dset(self):
        dset = self.get_dset()

        interface = self.config.DATASET['INTERFACE_TYPE']

        if interface == 'csv':
            all_dset = dset[
                [
                    *self.config.DATASET['X_COLS'],
                    self.config.DATASET['Y_COL']
                ]
            ].values

            all_x = all_dset[:, 0]
            all_y = all_dset[:, 1]

        elif interface == 'txt':
            delimiter = self.config.DATASET['DELIMITER']
            x_idx = self.config.DATASET['X_COLS']
            y_idx = self.config.DATASET['Y_COL']
            all_x = [np.array(i.split(delimiter))[*x_idx] for i in dset]
            all_y = [[i.split(delimiter)[y_idx] for i in dset]]

        elif interface == 'directory':
            return
        else:
            raise Exception('Not Supported Interface')

        train_x, valid_x, train_y, valid_y = train_test_split(
            all_x,
            all_y,
            test_size=self.config.DATASET['TEST_SPLIT_SIZE'],
            random_state=42
        )

        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y

    '''
		Gets all the dataset files and stores the indexes
	'''

    def get_dset(self):
        interface = self.config.DATASET['INTERFACE_TYPE']
        interface_path = self.config.DATASET['INTERFACE_PATH']
        if interface == 'csv':
            dset = pd.read_csv(interface_path)
        elif interface == 'directory':
            # Further splitting if self.config.dataset.interface_directories is true
            dset = os.listdir(interface_path)
        elif interface == 'txt':
            with open(interface_path, 'r') as f:
                dset = f.read()
                dset = [i for i in dset.split('\n') if len(i) != 0]

        return dset

    def get_indices(self):
        dset = self.get_dset()
        return np.arange(len(dset))

    def get_batch(self):
        idxs = self.shuffle()
        batch_idxs = np.random.choice(idxs, self.batch_size, False)
        return self.feed_function(dset, batch_idxs)

    def shuffle(self):
        idxs = self.get_indices()
        return np.random.shuffle(idxs)

    def augment(self):
        # Include Out Of The Box Augmentations For Image Data And Text Data And Structrured Data
        return

    def preprocess(self):
        # Just the resizing or padding or imputation, perform every time
        return
