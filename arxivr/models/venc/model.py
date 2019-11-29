# Imports
from keras.models import Sequential, Model
from keras.layers import Conv2D, ConvLSTM2D, InputLayer, UpSampling2D, MaxPooling2D, Input
from keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.initializers import glorot_uniform, RandomUniform
import keras.backend as K

import tensorflow as tf
import numpy as np

import cv2 as cv
from PIL import Image


class STEMPENC():
	def __init__(self):
		self.spatial_filters = 16
		self.spatial_filter_size = 7
		self.convlstm_filters = 64
		self.convlstm_filter_size = 7
		self.optical_flow_layers = 2
		self.optical_flow_layers_filters = 2
		self.optical_flow_layer_filter_size = 15
		self.huber_loss_init = Conv2D(filters=2, kernel_size=(3,3), trainable=False, use_bias=False)
		self.huber_loss_init.set_weights(np.array([
			[
				[0, 0, 0],
				[-0.5, 0, 0.5],
				[0, 0, 0]
			],[
				[0, -0.5, 0],
				[0, 0, 0],
				[0, 0.5, 0]
			]
		]))

		self.lr = 1e-4
		self.delta = 1e-3
		self.huber_weight = 1e-2
		self.inp_dims = (256, 256, 3)

		self.model = self.compose_model()

	def compose_model(self):
		inp = Input(shape=(self.inp_dims))

		self.spatial_encoder = self.get_spatial_encoder(inp)
		self.lstm_encoder = self.get_lstm()
		self.optical_flow = self.get_optical_flow()
		self.grid_gen = self.get_grid_generator()
		self.sampler = self.get_sampler()
		self.spatial_decoder = self.get_spatial_decoder()

		model = Model(inputs=inp, outputs=self.spatial_decoder.outputs)

		self.model = model

	def get_spatial_encoder(self, inp):
		x = Conv2D(
					input=self.inp_dims,
					filters=self.spatial_filters,
					kernel_size=self.spatial_filter_size,
					kernel_initializer=glorot_uniform,
					activation='tanh'
				)(inp)

		x = MaxPooling2D()(x)

		spatial_encoder = Model(inputs = inp, outputs = x)

		return spatial_encoder

	def get_lstm(self):
		encoding = self.spatial_encoder.outputs

		x = ConvLSTM2D(
				filters=self.convlstm_filters,
				kernel_size=self.convlstm_filter_size,
				kernel_initializer=RandomUniform(minval=-0.08, maxval=0.08)
			)(encoding)

		lstmModel = Model(inputs=encoding, outputs=x)

		return lstmModel

	def get_optical_flow(self):
		temporal_encoding = self.lstm_encoder.outputs

		x = Conv2D(
				input=(),
				filters=self.optical_flow_layers_filters,
				kernel_size=self.optical_flow_layer_filter_size,
				padding='same'
			)(temporal_encoding)

		x = Conv2D(
				filters=self.optical_flow_layers_filters,
				kernel_size=self.optical_flow_layer_filter_size,
				padding='same'
			)(x)

		x = Conv2D(
				filters=self.optical_flow_layers_filters,
				kernel_size=(1,1)
			)(x)

		optical_flow = Model(inputs=temporal_encoding, outputs=x)

		return optical_flow

	def get_grid_generator(self, h, w):
		grid = np.zeros((h, w, 3))
		# Create Grid Generator and Transform Matrix Generator Based On STN model
		return grid

	def get_sampler(self):
		sampler = None
		#TODO!!
		#
		return sampler


	def get_spatial_decoder(self):
		sampled = self.sampler.outputs
		x = UpSampling2D()(sampled)
		x = Conv2D(
				input=(),
				filters=self.spatial_filters,
				kernel_size=self.spatial_filter_size,
				kernel_initializer=glorot_uniform,
			)(x)

		spatial_decoder = Model(inputs=sampled, outputs=x)

		return spatial_decoder

	def huber_loss(self, del_t, delta=1.0):
		cond  = K.abs(del_t) < delta
		squared_loss = 0.5 * K.square(del_t)
		linear_loss  = delta * (K.abs(del_t) - 0.5 * delta)

		return np.where(cond, squared_loss, linear_loss)

	def loss_function(self, y_true, y_pred):
		del_t = self.huber_loss_init(self.optical_flow.outputs)
		huber_loss = self.huber_loss(del_t, self.delta)

		return K.square(y_pred - y_true) + (self.huber_weight * huber_loss)

	def compile(self):
		self.model.compile(optimizer='rmsprop', loss=self.loss_function, metrics=['acc'])

	def summary(self):
		self.model.summary()

	def train(self, train_x, train_y):
		def schedule(epoch, lr):
			if epoch%5 == 0:
				return lr - 0.9*lr
			return lr

		checkpoint = ModelCheckpoint('./checkpoints/', monitor='val_acc', verbose=0, save_best_only=True)
		earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=0)
		tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True)
		lrschedule = LearningRateScheduler(schedule, verbose=0)

		self.model.fit(train_x, train_y, epochs=20, batch_size=16 , callbacks=[checkpoint, earlystop, tensorboard, lrschedule])
