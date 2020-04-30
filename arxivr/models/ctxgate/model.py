from ..base import BASE
from .layers import GATEDCONV

import numpy as np

import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.applications.vgg16 import VGG16
from keras_contrib.layers import InstanceNormalization
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv2D, Conv2DTranspose, Activation, Dense, BatchNormalization, \
						Reshape, Input, Concatenate, Flatten, MaxPooling2D, multiply,    \
						LeakyReLU, Dropout, UpSampling2D, ZeroPadding2D, Lambda, Multiply

from keras.utils import plot_model

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

K.set_session(session)

class CTXGATE(BASE):
	def __init__(self, vars, model='ctxgate', inp_shape=(None, None, 3)):
		self.inp_shape = inp_shape
		self.model_name = model

		super(CTXGATE, self).__init__(vars)

	# Compose ResNet-50
	def compose_model(self):
		inp = Input(shape=self.inp_shape)

		x = GATEDCONV()(inp)
		x = BatchNormalization()(x)

		out = x

		model = Model(inputs=inp, outputs=out)
		model.compile(loss='categorical_crossentropy', optimizer=Adam())

		return model