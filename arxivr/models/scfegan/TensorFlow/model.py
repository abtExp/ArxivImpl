from .base import MODEL
from .layers import *
from .discriminator import DISCRIMINATOR
from .generator import GENERATOR
from ..utils.losses import *
from ..utils.utils import *

import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16

import tensorflow as tf

from functools import partial

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)

K.set_session(session)

class SCFEGAN():
	def __init__(self, config):
		self.model_name = 'scfegan'
		self.config = config
		self.generator = GENERATOR(config)
		self.discriminator = DISCRIMINATOR(config)
		self.feature_extractor = VGG16(input_shape=tuple(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE), include_top=False, weights='imagenet')
		self.feature_extractor.trainable = False

		self.DATA_LOADER = None

		# Graph For Discriminator

		# Discriminator Takes In Completed Or Real Image, Sketch, Color And Mask : (512, 512, 8)
		# Generator Takes In Incomplete Image, Mask, Sketch, Color, Noise : (512, 512, 9)

		gt_inp = Input(shape=(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[0], self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[1], 8))
		comp_inp = Input(shape=(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[0], self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[1], 8))

		real = self.discriminator.model(gt_inp)
		completed = self.discriminator.model(comp_inp)

		avgd = RandomWeightedAverage()([gt_inp[:,:,:,:3], comp_inp[:,:,:,:3]])

		avgd_out = self.discriminator.model(avgd)

		self.discriminator_model = Model(inputs=[gt_inp, comp_inp], outputs=[real, completed, avgd_out])

		partial_gp_loss = partial(
								gp_loss,
								averaged_samples=avgd
						)
		partial_gp_loss.__name__ = 'gradient_penalty'

		self.discriminator_model.compile(
										loss=['binary_crossentropy', 'binary_crossentropy', partial_gp_loss], 
										optimizer=Adam(),
										loss_weights=[1, 1, self.config.HYPERPARAMETERS.THETA]
									)

		# Graph For Generator Model
		gen_inp = Input(shape=(self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[0], self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[1], 9))

		out = self.generator.model(gen_inp)

		generator_loss = partial(
									generator_loss_function, 
									mask=gen_inp[:, :, :, 3], 
									feature_extractor=self.feature_extractor,
									config=self.config
								)

		self.generator_model = Model(inputs=gen_inp, outputs=out)

		self.generator_model.compile(loss=generator_loss, optimizer=Adam())

	def summary(self):
		self.generator.summary()
		self.discriminator.summary()

	def plot(self):
		self.generator.plot()
		self.discriminator.plot()

	def save(self):
		self.generator.save(self.config.CHECKPOINTS_PATH+'generator.hdf5')
		self.discriminator.save(self.config.CHECKPOINTS_PATH+'discriminator.hdf5')

	# Training Loop
	def train(self):
		for epoch in range(self.config.HYPERPARAMETERS.NUM_EPOCHS):
			for _ in range(self.config.HYPERPARAMETERS.NUM_DISC_ITERS):
				# Generate Batch Data
				img, incomplete_img, mask, sketch, color, noise, label = self.DATA_LOADER.generate_batch()
				gt = np.ones((8*8*256))
				comp = np.zeros((8*8*256))
				dummy = np.zeros((8*8*256))
				gt_inp = np.concat((img, mask, sketch, color), axis=0)
				gen_inp = np.concat((incomplete_img, mask, sketch, color, noise), axis=0)
				generated = self.generator.model(gen_inp)
				completed = complete_imgs(img, mask, generated)
				comp_inp = np.concat((completed, mask, sketch, color), axis=0)
				disc_loss = self.discriminator_model.train_on_batch([gt_inp, comp_inp], [gt, comp, dummy])
    
			_, incomplete_img, mask, sketch, color, noise, label = self.DATA_LOADER.generate_batch()
			g_inp = np.concat((incomplete_img, mask, sketch, color, noise), axis=0)
			gen_loss = self.generator_model.train_on_batch(g_inp)

			print("%d [D loss: %f] [G loss: %f]" % (epoch, disc_loss, gen_loss))

			if epoch % self.config.CONFIG_INFO.EVAL_EPOCH == 0:
				img, incomplete_img, mask, sketch, color, noise, label = self.DATA_LOADER.generate_batch()
				inp = np.concat((incomplete_img, mask, sketch, color, noise), axis=0)
				generated = self.generator.model(inp)
				completed = complete_imgs(img, mask, generated)
				cv2.imwrite('{}{}_{}_{}.jpg'.format(self.config.CONFIG_INFO.EVAL_DIR, epoch, disc_loss, gen_loss), completed)