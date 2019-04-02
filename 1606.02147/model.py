from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Conv2DTranspose, Input, Concatenate, Add, UpSampling2D, ZeroPadding2D, SpatialDropout2D, Permute, Reshape, Activation, Softmax
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.initializers import Ones
import keras.backend as K
from keras.utils import plot_model

import numpy as np
import tensorflow as tf

from os import listdir, path

from PIL import Image, ImageFile

import gc

class ENET():
	def __init__(self, vars):
		self.vars = vars
		self.compose_model()
		self.init_bootstrap_loaders()

	def init_bootstrap_loaders(self):
		self.train_loader = self.vars.DETECTOR_BOOTSTRAP_GENERATOR(self.vars.DETECTOR_ORIGINAL_BATCH_SIZE, self.vars.DETECTOR_BATCH_SIZE - self.vars.DETECTOR_ORIGINAL_BATCH_SIZE, self.vars)
		self.valid_loader = self.vars.DETECTOR_BOOTSTRAP_GENERATOR(self.vars.DETECTOR_ORIGINAL_BATCH_SIZE, self.vars.DETECTOR_BATCH_SIZE - self.vars.DETECTOR_ORIGINAL_BATCH_SIZE, self.vars, mode='valid')

	def compose_model(self):
		# Initial Block
		inp = Input(shape =self.vars.INP_SHAPE)
		x = Conv2D(filters=13, kernel_size=(3,3), strides=(2,2), padding='same')(inp)
		side = MaxPooling2D(pool_size=(2,2), strides=(2,2))(inp)
		x = Concatenate()([x,side])
		x = BatchNormalization()(x)
		x = PReLU(shared_axes=[1,2])(x)

		# block 1
		x = self.RDDNeck(x, 64, True, dilation=1, keep_probs=0.01)
		x = self.RDDNeck(x, 64, False, dilation=1, keep_probs=0.01)
		x = self.RDDNeck(x, 64, False, dilation=1, keep_probs=0.01)
		x = self.RDDNeck(x, 64, False, dilation=1, keep_probs=0.01)
		x = self.RDDNeck(x, 64, False, dilation=1, keep_probs=0.01)

		#block 2
		x = self.RDDNeck(x, 128, True, dilation=1)
		x = self.RDDNeck(x, 128, False, dilation=1)
		x = self.RDDNeck(x, 128, False, dilation=2)
		x = self.ASNeck(x, 128)
		x = self.RDDNeck(x, 128, False, dilation=4)
		x = self.RDDNeck(x, 128, False, dilation=1)
		x = self.RDDNeck(x, 128, False, dilation=8)
		x = self.ASNeck(x, 128)
		x = self.RDDNeck(x, 128, False, dilation=16)

		#block 3
		x = self.RDDNeck(x, 128, False, dilation=1)
		x = self.RDDNeck(x, 128, False, dilation=2)
		x = self.ASNeck(x, 128)
		x = self.RDDNeck(x, 128, False, dilation=4)
		x = self.RDDNeck(x, 128, False, dilation=1)
		x = self.RDDNeck(x, 128, False, dilation=8)
		x = self.ASNeck(x, 128)
		x = self.RDDNeck(x, 128, False, dilation=16)

		# block 4
		x = self.RDDNeck(x, 256, False, dilation=1)
		x = self.RDDNeck(x, 256, False, dilation=2)
		x = self.ASNeck(x, 256)
		x = self.RDDNeck(x, 256, False, dilation=4)
		x = self.RDDNeck(x, 256, False, dilation=1)
		x = self.RDDNeck(x, 256, False, dilation=8)
		x = self.ASNeck(x, 256)
		x = self.RDDNeck(x, 256, False, dilation=16)

		#block 4
		x = self.UBNeck(x, 64)
		x = self.RDDNeck(x, 64, False, dilation=1)
		x = self.RDDNeck(x, 64, False, dilation=1)

		#block 5
		x = self.UBNeck(x, 16)
		x = self.RDDNeck(x, 16, False, dilation=1)

		out = Conv2DTranspose(filters=self.vars.LOGO_NUM_CLASSES, kernel_size=(3,3), strides=(2,2), use_bias=False, output_padding=1, padding='same')(x)
		out = Reshape((self.vars.INP_SHAPE[0]*self.vars.INP_SHAPE[1], self.vars.LOGO_NUM_CLASSES))(out)
		out = Activation('softmax')(out)

		self.model = Model(inputs=inp, outputs=out)

		plot_model(self.model, './model_images/enet.png', show_shapes=True, show_layer_names=True)


	def RDDNeck(self, x, out_channels, down_flag, dilation=1, keep_probs=0.1, projection_ratio=4):

		inp = x


		if down_flag:
			stride = 2
			reduced_depth = int(int(inp.shape[-1]) // projection_ratio)
		else:
			stride = 1
			reduced_depth = int(out_channels // projection_ratio)

		# side branch
		x = Conv2D(filters=reduced_depth, kernel_size=(1, 1), strides=(1,1), use_bias=False, dilation_rate=1)(inp)
		x = BatchNormalization()(x)
		x = PReLU(shared_axes=[1,2])(x)

		x = ZeroPadding2D(padding=(dilation, dilation))(x)
		x = Conv2D(filters=reduced_depth, kernel_size=(3,3), strides=stride, use_bias=True, dilation_rate=dilation)(x)
		x = BatchNormalization()(x)
		x = PReLU(shared_axes=[1,2])(x)

		x = Conv2D(filters=out_channels, kernel_size=(1,1), strides=(1,1), use_bias=False, dilation_rate=1)(x)
		x = BatchNormalization()(x)

		x = SpatialDropout2D(keep_probs)(x)

		# main branch
		if down_flag:
			inp = MaxPooling2D(pool_size=(2,2), strides=2)(inp)
		if not inp.shape[-1] == out_channels:
			out_shape = out_channels - inp.shape[-1]
			inp = Permute((1,3,2))(inp)
			inp = ZeroPadding2D(padding=((0,0),(0,out_shape)))(inp)
			inp = Permute((1,3,2))(inp)

		x = Add()([x, inp])
		x = PReLU(shared_axes=[1,2])(x)

		return x

	def ASNeck(self, x, out_channels, projection_ratio=4):

		inp = x
		reduced_depth = int(int(inp.shape[-1])/projection_ratio)

		x = Conv2D(filters=reduced_depth, kernel_size=(1,1), strides=(1,1), use_bias=False)(inp)
		x = BatchNormalization()(x)
		x = PReLU(shared_axes=[1,2])(x)
		x = ZeroPadding2D(padding=(0,2))(x)
		x = Conv2D(filters=reduced_depth, kernel_size=(1,5), strides=(1,1), use_bias=False)(x)
		x = ZeroPadding2D(padding=(2,0))(x)
		x = Conv2D(filters=reduced_depth, kernel_size=(5,1), strides=(1,1), use_bias=False)(x)
		x = BatchNormalization()(x)
		x = PReLU(shared_axes=[1,2])(x)
		x = Conv2D(filters=out_channels, kernel_size=(1,1), strides=(1,1), use_bias=False)(x)

		x = SpatialDropout2D(0.1)(x)
		x = BatchNormalization()(x)

		if not inp.shape[-1] == out_channels:
			out_shape = out_channels - inp.shape[-1]
			inp = Permute((1,3,2))(inp)
			inp = ZeroPadding2D(padding=((0,0),(0,out_shape)))(inp)
			inp = Permute((1,3,2))(inp)

		x = Add()([x, inp])
		x = PReLU(shared_axes=[1,2])(x)

		return x

	def UBNeck(self, x, out_channels, projection_ratio=4):

		inp = x
		reduced_depth = int(int(inp.shape[-1])//projection_ratio)

		x = Conv2D(filters=reduced_depth, kernel_size=(1,1), use_bias=False)(inp)
		x = BatchNormalization()(x)
		x = PReLU(shared_axes=[1,2])(x)
		x = Conv2DTranspose(filters=reduced_depth, kernel_size=(3,3), strides=(2,2), use_bias=False, output_padding=1, padding='same')(x)
		x = BatchNormalization()(x)
		x = PReLU(shared_axes=[1,2])(x)

		x = Conv2D(filters=out_channels, kernel_size=(1,1), use_bias=False, padding='same')(x)
		x = BatchNormalization()(x)

		x = SpatialDropout2D(0.1)(x)

		inp = Conv2D(filters=out_channels, kernel_size=(1,1))(inp)
		inp = UpSampling2D(size=(2,2))(inp)

		x = Add()([x, inp])
		x = PReLU(shared_axes=[1,2])(x)

		return x


	def train(self, train_x, train_y):
		print('Training on batch...')
		self.model.train_on_batch(train_x, train_y)
		print('Training on current batch completed!')

	def summary(self):
		self.model.summary()

	def compile(self):
		self.model.compile(optimizer=Adam(lr=5e-4, decay=2e-3),
							loss='categorical_crossentropy')

	def bootstrap(self):
		directory = '{}/flikr/train'.format(self.vars.DETECTOR_TRAIN_DATA_PATH)
		print('Training...')
		# TODO apply the custom weighting scheme as used in the paper
		callbacks = self.vars.get_callbacks('enet')
		self.model.fit_generator(self.train_loader, steps_per_epoch=self.vars.DETECTOR_STEPS_PER_EPOCH, epochs=self.vars.DETECTOR_EPOCHS, callbacks=callbacks, workers=0, validation_data=self.valid_loader, initial_epoch=13, )

		print('Model Bootstrapping Completed!')
		self.model.save('./checkpoints/checkpoints_enet/final.hdf5')

	def detect(self, frame):
		return self.model.predict(frame)
