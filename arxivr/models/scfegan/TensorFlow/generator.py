from .base import MODEL

from tensorflow.keras.layers import Input, Concatenate, Activation
from tensorflow.keras.models import Model

from .layers import *

class GENERATOR(MODEL):
    def __init__(self, config):
        super(GENERATOR, self).__init__(config, 'generator')

    def compose_model(self):
        # Generator will take in the patched image, mask, sketch info, color_info and random noise
        inp = Input(shape=(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[0], self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[1], 9))
        cnum = 64
        x1, mask1 = GatedConv2D(inp, cnum, (7, 7), (2,2), use_lrn=False)
        x2, mask2 = GatedConv2D(x1, 2*cnum, (5, 5), (2, 2))
        x3, mask3 = GatedConv2D(x2, 4*cnum, (5, 5), (2, 2))
        x4, mask4 = GatedConv2D(x3, 8*cnum, (3, 3), (2, 2))
        x5, mask5 = GatedConv2D(x4, 8*cnum, (3, 3), (2, 2))
        x6, mask6 = GatedConv2D(x5, 8*cnum, (3, 3), (2, 2))
        x7, mask7 = GatedConv2D(x6, 8*cnum, (3, 3), (2, 2))

        x7, _ = GatedConv2D(x7, 8*cnum, (3, 3), (1, 1), dilation=2)
        x7, _ = GatedConv2D(x7, 8*cnum, (3, 3), (1, 1), dilation=4)
        x7, _ = GatedConv2D(x7, 8*cnum, (3, 3), (1, 1), dilation=8)
        x7, _ = GatedConv2D(x7, 8*cnum, (3, 3), (1, 1), dilation=16)

        x8, _ = GatedDeConv([self.config.HYPERPARAMETERS.TRAINING_BATCH_SIZE, int(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[0]/64), int(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[1]/64), 8*cnum])(x7)
        x8 = Concatenate(axis=0)([x6, x8])
        x8, mask8 = GatedConv2D(x8, 8*cnum, (3, 3), (1, 1))

        x9, _ = GatedDeConv([self.config.HYPERPARAMETERS.TRAINING_BATCH_SIZE, int(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[0]/32), int(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[1]/32), 8*cnum])(x8)
        x9 = Concatenate(axis=0)([x5, x9])
        x9, mask9 = GatedConv2D(x9, 8*cnum, (3, 3), (1, 1))

        x10, _ = GatedDeConv([self.config.HYPERPARAMETERS.TRAINING_BATCH_SIZE, int(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[0]/16), int(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[1]/16), 8*cnum])(x9)
        x10 = Concatenate(axis=0)([x4, x10])
        x10, mask10 = GatedConv2D(x10, 8*cnum, (3, 3), (1, 1))

        x11, _ = GatedDeConv([self.config.HYPERPARAMETERS.TRAINING_BATCH_SIZE, int(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[0]/8), int(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[1]/8), 4*cnum])(x10)
        x11 = Concatenate(axis=0)([x3, x11])
        x11, mask11 = GatedConv2D(x11, 4*cnum, (3, 3), (1, 1))

        x12, _ = GatedDeConv([self.config.HYPERPARAMETERS.TRAINING_BATCH_SIZE, int(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[0]/4), int(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[1]/4), 2*cnum])(x11)
        x12 = Concatenate(axis=0)([x2, x12])
        x12, mask12 = GatedConv2D(x12, 2*cnum, (3, 3), (1, 1))

        x13, _ = GatedDeConv([self.config.HYPERPARAMETERS.TRAINING_BATCH_SIZE, int(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[0]/2), int(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[1]/2), cnum])(x12)
        x13 = Concatenate(axis=0)([x1, x13])
        x13, mask13 = GatedConv2D(x13, cnum, (3, 3), (1, 1))

        x14, _ = GatedDeConv([self.config.HYPERPARAMETERS.TRAINING_BATCH_SIZE, int(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[0]), int(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[1]), 9])(x13)
        x14 = Concatenate(axis=0)([inp, x14])
        x14, mask14 = GatedConv2D(x14, 3, (3, 3), (1, 1))

        x14 = Activation('tanh')(x14)

        model = Model(inputs=inp, outputs=x14)

        return model