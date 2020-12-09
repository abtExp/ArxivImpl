from .base import MODEL
from .layers import *
from .discriminator import DISCRIMINATOR
from .generator import GENERATOR
from ..utils.losses import *
from ..utils.utils import *

import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
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
		self.feature_extractor = VGG16(
										input_shape=tuple(
												self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE
											), 
										include_top=False, 
										weights='imagenet'
								)
		self.feature_extractor.trainable = False

		self.DATA_LOADER = None

		# Graph For Discriminator
		gt_inp = Input(shape=(*self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[:2], 8))
		comp_inp = Input(shape=(*self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[:2], 8))
		avgd_inp = Input(shape=(*self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[:2], 8))

		real = self.discriminator.model(gt_inp)
		completed = self.discriminator.model(comp_inp)
		avgd_out = self.discriminator.model(avgd_inp)

		self.discriminator_model = Model(inputs=[gt_inp, comp_inp, avgd_inp], outputs=[real, completed, avgd_out])

		partial_gp_loss = partial(
								gp_loss,
								averaged_samples=avgd_inp
						)
		partial_gp_loss.__name__ = 'gradient_penalty'

		self.discriminator_model.compile(
										loss=[gt_loss, comp_loss, partial_gp_loss], 
										optimizer=Adam(),
										loss_weights=[1, 1, self.config.HYPERPARAMETERS.THETA]
									)

		# Graph For Generator Model
		self.discriminator.model.trainable = False
		gt_img_inp = Input(shape=(*self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[:2], 3))
		incomp_img_inp = Input(shape=(*self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[:2], 3))

		mask = Input(shape=(*self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[:2], 1))
		noise = Input(shape=(*self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[:2], 1))
		color_sketch = Input(shape=(*self.config.MODEL.MODEL_PARAMS.INPUT_SHAPE[:2], 4))

		generator_input = Concatenate()([incomp_img_inp, mask, noise, color_sketch])
		generated = self.generator.model(generator_input)

		completed = COMPLETE_IMAGE_LAYER()([gt_img_inp, mask, generated])
		disc_inp_comp = Concatenate()([completed, mask, color_sketch])
		disc_out_comp = self.discriminator.model(disc_inp_comp)
		disc_inp_gt = Concatenate()([gt_img_inp, mask, color_sketch])
		disc_out_gt = self.discriminator.model(disc_inp_gt)

		generator_loss = partial(
							generator_loss_function, 
							mask=mask, 
							feature_extractor=self.feature_extractor,
							config=self.config
						)

		self.generator_model = Model(
								inputs=[
										gt_img_inp, 
										incomp_img_inp, 
										mask, 
										noise, 
										color_sketch
									], 
								outputs=[
										generated, 
										disc_out_comp, 
										disc_out_gt
									]
							)
		self.generator_model.compile(
								loss=[
										generator_loss, 
										gsn_loss, 
										add_term_loss
								], 
								optimizer=Adam(), 
								loss_weights=[
										1, 
										self.config.HYPERPARAMETERS.BETA,
										self.config.HYPERPARAMETERS.EPS
								]
							)

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
				avgd_img = random_weighted_average(img, completed)
				avgd_inp = np.concat((avgd_img, mask, sketch, color), axis=0)
				disc_loss = self.discriminator_model.train_on_batch([gt_inp, comp_inp, avgd_inp], [gt, comp, dummy])
    
			img, incomplete_img, mask, sketch, color, noise, label = self.DATA_LOADER.generate_batch()
			gen_loss = self.generator_model.train_on_batch([img, incomplete_img, mask, noise, np.concat((color, sketch))])

			print("%d [D loss: %f] [G loss: %f]" % (epoch, disc_loss, gen_loss))

			if epoch % self.config.CONFIG_INFO.EVAL_EPOCH == 0:
				img, incomplete_img, mask, sketch, color, noise, label = self.DATA_LOADER.generate_batch()
				inp = np.concat((incomplete_img, mask, sketch, color, noise), axis=0)
				generated = self.generator.model(inp)
				completed = complete_imgs(img, mask, generated)
				cv2.imwrite('{}{}_{}_{}.jpg'.format(self.config.CONFIG_INFO.EVAL_DIR, epoch, disc_loss, gen_loss), completed)