from .base import MODEL

from .layers import *

from tensorflow.keras.layers import Input, Flatten, ZeroPadding2D, Conv2D
from tensorflow.keras.models import Model


class DISCRIMINATOR(MODEL):
    def __init__(self, config):
        super(DISCRIMINATOR, self).__init__(config, 'discriminator')

    def compose_model(self):
        # Input Is Generated Image(Completed) Or Ground Truth Image, Mask, Sketch And Color Info
        inp = Input(shape=(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[0], self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[1], 8))
        x = SpectralNormalization(Conv2D(filters=64, kernel_size=(3, 3), strides=(2,2), activation='relu'))(inp)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = SpectralNormalization(Conv2D(filters=128, kernel_size=(3, 3), strides=(2,2), activation='relu'))(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = SpectralNormalization(Conv2D(filters=256, kernel_size=(3, 3), strides=(2,2), activation='relu'))(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = SpectralNormalization(Conv2D(filters=256, kernel_size=(3, 3), strides=(2,2), activation='relu'))(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = SpectralNormalization(Conv2D(filters=256, kernel_size=(3, 3), strides=(2,2), activation='relu'))(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = SpectralNormalization(Conv2D(filters=256, kernel_size=(3, 3), strides=(2,2), activation='sigmoid'))(x)
        x = Flatten()(x)

        model = Model(inputs=inp, outputs=x)

        return model