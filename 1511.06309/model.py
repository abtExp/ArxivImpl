# Imports
from keras.models import Sequential, Model
from keras.layers import Conv2d, ConvLSTM2D, InputLayer, UpSampling2D
from keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.initializers import glorot_uniform, RandomUniform

import utils

import cv2 as cv
from PIL import Image


# Defining Class
class AutoEnc():
	def __init__(self):
		self.spatial_filters = 16
		self.spatial_filter_size = 7
		self.convlstm_filters = 64
		self.convlstm_filter_size = 7
		self.optical_flow_layers = 2
		self.optical_flow_layers_filters = [2, 2]
		self.optical_flow_layer_filter_size = [15,15]

		self.lr = 1e-4
		self.delta = 1e-3
		self.inp_dims = (256, 256, 3)

		self.model = self.compose_model()
	
	def compose_model(self):
		self.spatial_encoder = self.get_spatial_encoder()
		self.lstm_encoder = self.get_lstm()
		self.optical_flow = self.get_optical_flow()
		self.grid_gen = self.get_grid_generator()
		self.sampler = self.get_sampler()
		self.spatial_decoder = self.get_spatial_decoder()

		model = Model()
		

		return model

	def compile(self):
		self.model.compile(optimizer='rmsprop', loss=self.loss_function, metrics=['acc'])

	def get_spatial_encoder(self):
		spatial_encoder = Sequential()
		spatial_encoder.add(Conv2d(input=self.inp_dims, filters=self.spatial_filters, kernel_size=self.spatial_filter_size, kernel_initializer=glorot_uniform))

		return spatial_encoder
	
	def get_spatial_decoder(self):
		spatial_decoder = Sequential()
		spatial_decoder.add(Conv2d(input=(), filters=self.spatial_filters, kernel_size=self.spatial_filter_size, kernel_initializer=glorot_uniform))
		spatial_decoder.add(UpSampling2D())

		return spatial_decoder
	
	def get_lstm(self):
		lstmModel = Sequential()
		lstmModel.add(ConvLSTM2D(input=(), filters=self.convlstm_filters, kernel_size=self.convlstm_filter_size, kernel_initializer=RandomUniform(minval=-0.08, maxval=0.08)))

		return lstmModel

	def get_optical_flow(self):
		optical_flow = Sequential()
		
		for i in range(self.optical_flow_layers):
			optical_flow.add(Conv2d(filters=self.optical_flow_layers_filters[i], kernel_size=self.optical_flow_layer_filter_size[i]))
		
		optical_flow.add(Conv2d(kernel_size=(1,1)))
		optical_flow.add(utils.HuberLossLayer())

		return optical_flow

	def get_grid_generator(self):
		gg = None

		return gg
	
	def get_sampler(self):
		sampler = None
		
		return sampler

	def train(self, train_x, train_y):
		def schedule(epoch, lr):
			if epoch%5 == 0:
				return lr - 0.9*lr

			return lr

		checkpoint = ModelCheckpoint('./checkpoints/', monitor='val_acc', verbose=0, save_best_only=True)
		earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=0)
		tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True)
		lrschedule = LearningRateScheduler(schedule, verbose=0)

		self.model.fit(train_x, train_y, callbacks=[checkpoint, earlystop, tensorboard, lrschedule])
