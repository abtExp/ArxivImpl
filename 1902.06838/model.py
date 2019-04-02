import keras.backend as K
from keras.layers import Conv2D, Conv2DTranspose, Activation, Dense, BatchNormalization, Reshape, Input, Concatenate, Flatten, MaxPooling2D, multiply, LeakyReLU, Dropout, UpSampling2D, ZeroPadding2D, Lambda, Multiply
from keras.models import Model
from keras_contrib.layers import InstanceNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.applications.vgg16 import VGG16
from keras.utils import plot_model

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
import cv2

from os import listdir
from os.path import join

class SCFEGAN():
	def __init__(self, vars):
		self.vars = vars
		self.LRNLayer = self.vars.LRN_LAYER
		self.GatedDeConv = self.vars.GATED_DE_CONV
		self.masks = None
		self.compose_model()

	def compose_model(self):
		self.discriminator = self.get_discriminator()
		self.generator = self.get_generator()
		self.feature_extractor = VGG16(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
		self.feature_extractor.trainable = False

		self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam())
		# self.generator.compile(loss=self.generator_loss_function, optimizer=Adam())

		inp = Input(shape=(self.vars.INP_SHAPE[0], self.vars.INP_SHAPE[1], 9))

		gen, _ = self.generator(inp)

		# comp = self.complete_imgs(inp[0], inp[1], gen)

		self.discriminator.trainable=False

		res = self.discriminator(gen)

		self.model = Model(inputs=inp, outputs=res)

		self.model.summary()

		self.model.compile(loss='binary_crossentropy', optimizer=Adam())

		# plot_model(self.model, './scfegan.png', show_shapes=True)

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

		x8, _ = self.GatedDeConv2D(x7, [self.vars.SCFEGAN_BATCH_SIZE, int(self.vars.INP_SHAPE[0]/64), int(self.vars.INP_SHAPE[1]/64), 8*cnum])
		x8 = Concatenate(axis=0)([x6, x8])
		x8, mask8 = self.GatedConv2D(x8, 8*cnum, (3, 3), (1, 1))

		x9, _ = self.GatedDeConv2D(x8, [self.vars.SCFEGAN_BATCH_SIZE, int(self.vars.INP_SHAPE[0]/32), int(self.vars.INP_SHAPE[1]/32), 8*cnum])
		x9 = Concatenate(axis=0)([x5, x9])
		x9, mask9 = self.GatedConv2D(x9, 8*cnum, (3, 3), (1, 1))

		x10, _ = self.GatedDeConv2D(x9, [self.vars.SCFEGAN_BATCH_SIZE, int(self.vars.INP_SHAPE[0]/16), int(self.vars.INP_SHAPE[1]/16), 8*cnum])
		x10 = Concatenate(axis=0)([x4, x10])
		x10, mask10 = self.GatedConv2D(x10, 8*cnum, (3, 3), (1, 1))

		x11, _ = self.GatedDeConv2D(x10, [self.vars.SCFEGAN_BATCH_SIZE, int(self.vars.INP_SHAPE[0]/8), int(self.vars.INP_SHAPE[1]/8), 4*cnum])
		x11 = Concatenate(axis=0)([x3, x11])
		x11, mask11 = self.GatedConv2D(x11, 4*cnum, (3, 3), (1, 1))

		x12, _ = self.GatedDeConv2D(x11, [self.vars.SCFEGAN_BATCH_SIZE, int(self.vars.INP_SHAPE[0]/4), int(self.vars.INP_SHAPE[1]/4), 2*cnum])
		x12 = Concatenate(axis=0)([x2, x12])
		x12, mask12 = self.GatedConv2D(x12, 2*cnum, (3, 3), (1, 1))

		x13, _ = self.GatedDeConv2D(x12, [self.vars.SCFEGAN_BATCH_SIZE, int(self.vars.INP_SHAPE[0]/2), int(self.vars.INP_SHAPE[1]/2), cnum])
		x13 = Concatenate(axis=0)([x1, x13])
		x13, mask13 = self.GatedConv2D(x13, cnum, (3, 3), (1, 1))

		x14, _ = self.GatedDeConv2D(x13, [self.vars.SCFEGAN_BATCH_SIZE, int(self.vars.INP_SHAPE[0]), int(self.vars.INP_SHAPE[1]), 9])
		x14 = Concatenate(axis=0)([inp, x14])
		x14, mask14 = self.GatedConv2D(x14, 3, (3, 3), (1, 1))

		x14 = Activation('tanh')(x14)

		model = Model(inputs=inp, outputs=[x14, mask14])

		model.summary()

		# plot_model(model, './scfegan_gen.png', show_shapes=True)
		return model

	def GatedConv2D(self, x, filters, kernel_size, strides, dilation=1, activation='leaky_relu', use_lrn=True):
		inp = x
		x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation, padding='same')(x)
		if use_lrn:
			x = LRNLayer()(x)

		if activation == 'leaky_relu':
			x = LeakyReLU()(x)
		else:
			x = Activation(activation)('x')

		g = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', dilation_rate=dilation)(inp)
		g = Activation('sigmoid')(g)

		x = multiply([x, g])

		return x, g

	def GatedDeConv2D(self, x, out_shape, kernel_size=(5, 5), strides=(2, 2), std_dev=0.02):
		return GatedDeConv(out_shape, kernel_size, strides, std_dev)(x)

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

		model.summary()

		# plot_model(model, './scfegan_dis.png', show_shapes=True)
		return model

	def complete_imgs(self, images, masks, generated):
		completed_images = np.zeros(np.shape(images))

		patches = generated * masks

		reversed_mask = np.logical_not(masks, dtype=np.int)
		completion = images * reversed_mask

		completed_images = np.add(patches, completion)

		return completed_images

	def generator_loss_function(self, images, generated, masks):
		cmp_images = self.complete_imgs(np.array(images), masks, generated)

		ppxl_loss = self.per_pixel_loss(images, generated, masks, self.vars.SCFEGAN_ALPHA)
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


	def discriminator_loss_function(self, images, completed, masks):
		loss = (1 - images) + (1 + completed) + self.vars.SCFEGAN_THETA * self.gp_loss(masks)
		return loss


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

	def per_pixel_loss(self, images, generated, masks, alpha):
		nf = np.prod(np.shape(images[0]))

		t1 = (np.multiply(masks, np.subtract(generated, images)))/nf
		t2 = (np.multiply((1 - masks), np.subtract(generated, images)))/nf

		ppl = t1 + alpha * t2

		return ppl

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

	# TODO ( Test )
	def total_variation_loss(self, completed):
		tvl_row = [((completed[:,i+1, j, :] - completed[:,i,j, :])/np.shape(completed)[-1]) \
				for i in range(np.shape(completed[1])) for j in range(np.shape(completed)[2])]

		tvl_col = [((completed[:,i, j+1, :] - completed[:,i,j, :])/np.shape(completed)[-1]) \
		for i in range(np.shape(completed[1])) for j in range(np.shape(completed)[2])]

		tvl = tvl_row + tvl_col

		return tvl

	def gp_loss(self, masks):
		data_point = np.random.rand(*self.vars.INP_SHAPE)
		gpl = (np.multiply(self.discriminator_model(data_point), masks)**2 - 1)**2
		return gpl

	def gsn_loss(self, completed):
		return -1 * self.discriminator_model(completed)

	def train(self):
		for e in range(self.vars.SCFEGAN_TRAIN_EPOCHS):
			inp, images = self.vars.SCFEGAN_DATA_LOADER(self.vars)

			masks = inp[1]

			self.masks = masks

			valid = np.ones((self.vars.SCFEGAN_BATCH_SIZE, *self.vars.SCFEGAN_DISC_OP_SIZE))
			fakes = np.zeros((self.vars.SCFEGAN_BATCH_SIZE, *self.vars.SCFEGAN_DISC_OP_SIZE))

			gen_imgs, _ = self.generator.predict(inp)

			cmp_images = self.complete_imgs(images, masks, gen_imgs)

			d_gt_loss = self.discriminator.train_on_batch(images, valid)
			d_cmp_loss = self.discriminator.train_on_batch(cmp_images, fakes)
			# generator_loss = self.generator.train_on_batch(inp, images)

			d_gt = self.discriminator(images)

			d_cmp = self.discriminator(cmp_images)

			discriminator_loss = self.discriminator_loss_function(d_gt, d_cmp, masks)

			generator_loss = self.generator_loss_function(images, gen_imgs, masks)

			total_loss = self.model.train_on_batch(inp, valid)

			print('GENERATOR_LOSS : {}, DISCRIMINATOR_LOSS : {}, TOTAL_LOSS : {}'.format(generator_loss, discriminator_loss, total_loss))

			if e % 10 == 0:
				self.sample(e)

	def sample(self, epoch):
		r, c = 2, 5
		inps, _ = self.vars.SCFEGAN_DATA_LOADER(1)
		inp_imgs, inp_masks, inp_sketches, inp_colors, inp_noise = inps
		op = self.generator.predict(inps)
		op = op*0.5 + 0.5

		fig, axs = plt.subplots(r, c)
		for j in range(c):
			axs[0, j].imshow(inps[0,j])

		axs[1,0].imshow(op)
		fig.savefig('./test/test_scfegan/{}.png'.format(epoch))
		plt.close()