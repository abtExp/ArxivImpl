import sys
sys.path.insert(0, './models')
sys.path.insert(0, './utils')

from base import BASE
from scfegan_utils import *
from data_loaders import scfegan_data_loader

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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

K.set_session(session)


class SCFEGAN(BASE):
	def __init__(self, vars, model='scfegan', inp_shape=(None, None, 1)):
		self.inp_shape = inp_shape
		self.model_name = model
		self.LRNLayer = LRNLayer
		self.GatedDeConv = GatedDeConv

		super(SCFEGAN, self).__init__(vars)

		self.vars.DATA_LOADER = self.vars.SCFEGAN_DATA_LOADER

	def compose_model(self):
		self.discriminator = self.get_discriminator()
		self.generator = self.get_generator()
		self.feature_extractor = VGG16(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
		self.feature_extractor.trainable = False

		self.discriminator.compile(loss=self.disc_loss, optimizer=Adam())
		self.generator.compile(loss=self.gen_loss, optimizer=Adam())

	def get_generator(self):
		# Generator will take in the patched image, mask, sketch info, color_info and random noise
		inp = Input(shape=(self.vars.INP_SHAPE[0], self.vars.INP_SHAPE[1], 9))
		cnum = 64
		x1, mask1 = self.GatedConv2D(inp, cnum, (7, 7), (2,2), use_lrn=False)
		x2, mask2 = self.GatedConv2D(x1, 2*cnum, (5, 5), (2, 2))
		x3, mask3 = self.GatedConv2D(x2, 4*cnum, (5, 5), (2, 2))
		x4, mask4 = self.GatedConv2D(x3, 8*cnum, (3, 3), (2, 2))
		x5, mask5 = self.GatedConv2D(x4, 8*cnum, (3, 3), (2, 2))
		x6, mask6 = self.GatedConv2D(x5, 8*cnum, (3, 3), (2, 2))
		x7, mask7 = self.GatedConv2D(x6, 8*cnum, (3, 3), (2, 2))

		x7, _ = self.GatedConv2D(x7, 8*cnum, (3, 3), (1, 1), dilation=2)
		x7, _ = self.GatedConv2D(x7, 8*cnum, (3, 3), (1, 1), dilation=4)
		x7, _ = self.GatedConv2D(x7, 8*cnum, (3, 3), (1, 1), dilation=8)
		x7, _ = self.GatedConv2D(x7, 8*cnum, (3, 3), (1, 1), dilation=16)

		x8, _ = self.GatedDeConv2D(x7, [self.vars.TRAIN_BATCH_SIZE, int(self.vars.INP_SHAPE[0]/64), int(self.vars.INP_SHAPE[1]/64), 8*cnum])
		x8 = Concatenate(axis=0)([x6, x8])
		x8, mask8 = self.GatedConv2D(x8, 8*cnum, (3, 3), (1, 1))

		x9, _ = self.GatedDeConv2D(x8, [self.vars.TRAIN_BATCH_SIZE, int(self.vars.INP_SHAPE[0]/32), int(self.vars.INP_SHAPE[1]/32), 8*cnum])
		x9 = Concatenate(axis=0)([x5, x9])
		x9, mask9 = self.GatedConv2D(x9, 8*cnum, (3, 3), (1, 1))

		x10, _ = self.GatedDeConv2D(x9, [self.vars.TRAIN_BATCH_SIZE, int(self.vars.INP_SHAPE[0]/16), int(self.vars.INP_SHAPE[1]/16), 8*cnum])
		x10 = Concatenate(axis=0)([x4, x10])
		x10, mask10 = self.GatedConv2D(x10, 8*cnum, (3, 3), (1, 1))

		x11, _ = self.GatedDeConv2D(x10, [self.vars.TRAIN_BATCH_SIZE, int(self.vars.INP_SHAPE[0]/8), int(self.vars.INP_SHAPE[1]/8), 4*cnum])
		x11 = Concatenate(axis=0)([x3, x11])
		x11, mask11 = self.GatedConv2D(x11, 4*cnum, (3, 3), (1, 1))

		x12, _ = self.GatedDeConv2D(x11, [self.vars.TRAIN_BATCH_SIZE, int(self.vars.INP_SHAPE[0]/4), int(self.vars.INP_SHAPE[1]/4), 2*cnum])
		x12 = Concatenate(axis=0)([x2, x12])
		x12, mask12 = self.GatedConv2D(x12, 2*cnum, (3, 3), (1, 1))

		x13, _ = self.GatedDeConv2D(x12, [self.vars.TRAIN_BATCH_SIZE, int(self.vars.INP_SHAPE[0]/2), int(self.vars.INP_SHAPE[1]/2), cnum])
		x13 = Concatenate(axis=0)([x1, x13])
		x13, mask13 = self.GatedConv2D(x13, cnum, (3, 3), (1, 1))

		x14, _ = self.GatedDeConv2D(x13, [self.vars.TRAIN_BATCH_SIZE, int(self.vars.INP_SHAPE[0]), int(self.vars.INP_SHAPE[1]), 9])
		x14 = Concatenate(axis=0)([inp, x14])
		x14, mask14 = self.GatedConv2D(x14, 3, (3, 3), (1, 1))

		x14 = Activation('tanh')(x14)

		model = Model(inputs=inp, outputs=x14)

		return model

	def get_discriminator(self):
		inp = Input(shape=self.vars.INP_SHAPE)
		x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(inp)
		x = ZeroPadding2D(padding=(1, 1))(x)
		x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
		x = ZeroPadding2D(padding=(1, 1))(x)
		x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
		x = ZeroPadding2D(padding=(1, 1))(x)
		x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
		x = ZeroPadding2D(padding=(1, 1))(x)
		x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
		x = ZeroPadding2D(padding=(1, 1))(x)
		x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)

		model = Model(inputs=inp, outputs=x)

		return model

	def GatedConv2D(self, x, filters, kernel_size, strides, dilation=1, activation='leaky_relu', use_lrn=True):
		inp = x
		x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation, padding='same')(x)
		if use_lrn:
			x = self.LRNLayer()(x)

		if activation == 'leaky_relu':
			x = LeakyReLU()(x)

		else:
			x = Activation(activation)('x')

		g = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', dilation_rate=dilation)(inp)
		g = Activation('sigmoid')(g)

		x = multiply([x, g])

		return x, g

	def GatedDeConv2D(self, x, out_shape, kernel_size=(5, 5), strides=(2, 2), std_dev=0.02):
		return self.GatedDeConv(out_shape, kernel_size, strides, std_dev)(x)

	def complete_imgs(self, images, masks, generated):
		completed_images = np.zeros(np.shape(images))

		patches = generated * masks

		reversed_mask = np.logical_not(masks, dtype=np.int)
		completion = images * reversed_mask

		completed_images = np.add(patches, completion)

		return completed_images

	def gen_loss(self, y_true, y_pred):
		return y_true - y_pred

	def disc_loss(self, y_true, y_pred):
		return y_true - y_pred

	# ! Test
	def generator_loss_function(self, images, generated):

		cmp_images = self.complete_imgs(np.array(images), self.masks, generated)

		ppxl_loss = self.per_pixel_loss(images, generated, self.masks, self.vars.SCFEGAN_ALPHA)
		perc_loss = self.perceptual_loss(images, generated, cmp_images)
		g_sn_loss = self.gsn_loss(cmp_images)
		sg_loss = self.style_loss(generated)
		sc_loss = self.style_loss(cmp_images)
		tv_loss = self.total_variation_loss(cmp_images)
		add_term = K.square(self.discriminator(images))

		g_loss = ppxl_loss + (self.vars.SCFEGAN_SIGMA * perc_loss)\
				+ (self.vars.SCFEGAN_BETA * g_sn_loss) + (self.vars.SCFEGAN_GAMMA *\
				(sg_loss + sc_loss)) + (self.vars.SCFEGAN_V * tv_loss) + add_term

		return g_loss

	# ! Test
	def discriminator_loss_function(self, images, completed):
		loss = (1 - images) + (1 + completed) + self.vars.SCFEGAN_THETA * self.gp_loss(self.masks)
		return loss

	# ! Test
	def extract_features(self, x):
		feature_extractor = self.feature_extractor
		activations = feature_extractor(x)
		nf = []
		outputs = []

		for layer in feature_extractor.layers:
			if layer.name in ["block1_pool", "block2_pool", "block3_pool"]:
				nf.append(np.prod(K.shape(layer.get_weights())))
				outputs.append(layer.output)

		return outputs, nf

	# ! Test
	def per_pixel_loss(self, images, generated, masks, alpha):
		nf = np.prod(np.shape(images[0]))

		t1 = (np.multiply(masks, np.subtract(generated, images)))/nf
		t2 = (np.multiply((1 - masks), np.subtract(generated, images)))/nf

		ppl = t1 + alpha * t2

		return ppl

	# ! Test
	def perceptual_loss(self, images, generated, completed):
		gt_activs, nf = self.extract_features(images)
		gen_activs, _ = self.extract_features(generated)
		cmp_activs, _ = self.extract_features(completed)

		t1 = np.sum((np.subtract(gen_activs, gt_activs))/nf)
		t2 = np.sum((np.subtract(cmp_activs, gt_activs))/nf)

		pl = t1 + t2
		return pl

	def style_loss(self, gen, gt):
		gt_features, nc = self.extract_features(gt)
		gen_features, _ = self.extract_features(gen)

		per_layer_features = nc ** 2

		t1 = np.dot(gen_features, (self.vars.SCFEGAN_THETA - gen_features))
		t2 = np.dot(gt_features, (self.vars.SCFEGAN_THETA - gt_features))

		sl = np.sum((t1 - t2)/per_layer_features)

		return sl

	# ! Test
	def total_variation_loss(self, completed):
		tvl_row = [((completed[:,i+1, j, :] - completed[:,i,j, :])/np.size(completed)) \
				for i in range(np.shape(completed[1])) for j in range(np.shape(completed)[2])]

		tvl_col = [((completed[:,i, j+1, :] - completed[:,i,j, :])/np.size(completed)) \
		for i in range(np.shape(completed[1])) for j in range(np.shape(completed)[2])]

		tvl = tvl_row + tvl_col

		return tvl

	# ! Test
	def gp_loss(self, masks):
		data_point = np.random.rand(*self.vars.INP_SHAPE)
		gpl = (np.sqrt(np.multiply(grad(self.discriminator_model(data_point)), masks)) - 1)**2
		return gpl

	def gsn_loss(self, completed):
		return -1 * self.discriminator_model(completed)

	def train(self):
		for e in range(self.vars.SCFEGAN_TRAIN_EPOCHS):

			x, y = scfegan_data_loader(self.vars)
			print('Loaded Generator Data')

			masks = np.array(x[1])

			self.masks = masks

			gen_loss = self.generator.train_on_batch(x, y)

			print('Trained Generator On 1 Batch')

			x_, y_ = scfegan_data_loader(self.vars)

			print('Loaded Discriminator Data')

			generated = self.generator.predict(x_)

			print('Got Generator Predictions')

			self.masks = np.array(x_[0])

			completed = self.complete_imgs(y_, x_[0], x_)

			valid = np.ones(self.vars.SCFEGAN_DISC_OP_SHAPE)
			fakes = np.zeros(self.vars.SCFEGAN_DISC_OP_SHAPE)

			disc_loss_1 = self.discriminator.train_on_batch(y_, valid)
			print('Trained On Valid Batch')
			disc_loss_2 = self.discriminator.train_on_batch(completed, fakes)
			print('Trained On Fake Batch')

			print('GENERATOR LOSS : {}, DISCRIMINATOR_LOSS : {}'.format(gen_loss, (disc_loss_1+disc_loss_2)/2))

			if e % self.vars.LOG_EPOCH == 0:
				print('SAVING : {}'.format(e))
				self.generator.save('./checkpoints/scfegan/generator_{}_{}.hdf5'.format(e, gen_loss))
				self.discriminator.save('./checkpoints/scfegan/discriminator_{}_{}.hdf5'.format(e, (disc_loss_1+disc_loss_2)/2))
