from .base import MODEL
from .layers import *
from .discriminator import DISCRIMINATOR
from .generator import GENERATOR

import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16

import tensorflow as tf

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)

K.set_session(session)

class SCFEGAN():
	def __init__(self, config):
		self.model_name = 'scfegan'

		self.generator = GENERATOR(config)
		self.discriminator = DISCRIMINATOR(config)
		self.feature_extractor = VGG16(input_shape=(512, 512, 3), include_top=False, weights='imagenet')
		self.feature_extractor.trainable = False

		super(SCFEGAN, self).__init__(config)

		self.config.DATA_LOADER = None

	def compose_model(self):
		self.discriminator.compile(loss=self.disc_loss, optimizer=Adam())
		self.generator.compile(loss=self.gen_loss, optimizer=Adam())

	def summary(self):
		self.generator.summary()
		self.discriminator.summary()

	def plot(self):
		self.generator.plot()
		self.discriminator.plot()

	def save(self):
		self.generator.save(self.config.CHECKPOINTS_PATH+'generator.hdf5')
		self.discriminator.save(self.config.CHECKPOINTS_PATH+'discriminator.hdf5')

	# Create The Logger And The Progress Bars For Tracking Performance
	# ! UNDER CONSTRUCTION !
	def train_(self):
		for e in range(self.config.SCFEGAN_TRAIN_EPOCHS):

			x, y = scfegan_data_loader(self.config)

			self.masks = np.array(x[1])

			generator_loss = self.generator.train_on_batch(x, y)

			x_, y_ = scfegan_data_loader(self.config)

			self.masks = np.array(x_[1])

			generated = self.generator(x_)

			completed = self.complete_imgs(y_, self.masks, generated)

			# Discriminator expects the image, the masks, sketch, and the
			# channels as input
			y_ = np.array(x[:3])

			x_ = self.discriminator(completed)

			discriminator_loss = self.discriminator.train_on_batch(x_, y_)

			print('generator_loss : {.%2f} discriminator_loss : {%2f}'.format(generator_loss, discriminator_loss))

			if e%self.config.LOG_EPOCH == 0:
				self.save()