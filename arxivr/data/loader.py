import os
import numpy as np
import pandas as pd
import tensorflow as tf

from .feeder import FEEDER

class LOADER(tf.keras.utils.Sequence):
    def __init__(self, config, feed_function, mode='train'):
        self.config = config
        self.feeder = FEEDER(config, feed_function)

        if mode == 'train':
            self.batch_size = self.config.TRAIN_BATCH_SIZE
        elif mode == 'valid':
            self.batch_size = self.config.VALID_BATCH_SIZE

        self.all_indices = self.feeder.get_indices()

        self.on_epoch_end()

    # Overwrite if a special kind of loader is needed, like one with synthetic data generation
    def __getitem__(self, index):
        indices = self.all_indices[0][index *
                                    self.batch_size:(index+1)*self.batch_size]

        features = [self.all_indices[0][i] for i in indices]
        labels = [self.all_indices[1][i] for i in indices]

        params = {
            'features': features,
            'labels': labels
        }

        x, y = self.__data_generation(params)

        return x, y

    def __len__(self):
        length = int(np.floor(self.dataset_len / self.batch_size))
        return length

    def __data_generation(self, params):
        x, y = self.loader(self.config, self.mode, **params)

        return x, y

    def on_epoch_end(self):
        for i in range(self.all_indices):
            index = np.arrange(len(self.all_indices[i]))
            np.random.shuffle(index)
            self.indices.append(index)
