from keras.layers import Layer, Conv2D, Conv2DTranspose, Reshape, Multiply, LeakyReLU, Activation, multiply, BatchNormalization
from keras.initializers import RandomNormal
import keras.backend as K
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import cv2

import os

from os import listdir

import utils

class LRNLayer(Layer):
	def __init__(self, alpha=1e-4, beta=0.75, k=2, n=5):
		super(LRNLayer, self).__init__()
		self.alpha = alpha
		self.beta = beta
		self.k = k
		self.n = n

	def call(self, x):
		op = []
		nc = np.shape(x)[-1]
		for i in range(nc):
			sq = K.sum((x[:,:,:,max(0, int(i-self.n/2)):min(nc-1, i+int(self.n/2))+1]) ** 2)
			op.append(x[:,:,:,i]/((self.k + self.alpha * sq) ** self.beta))

		op = tf.convert_to_tensor(op)

		op = tf.transpose(op, perm=[1,2,3,0])

		op_shape = self.compute_output_shape(np.shape(x))

		op._keras_shape = op_shape

		return op

	def compute_output_shape(self, input_shape):
		return input_shape

	def compute_mask(self, input, input_mask):
		return 1*[None]

class GatedDeConv(Layer):
	def __init__(self, out_shape, kernel_size, strides, std_dev):
		super(GatedDeConv, self).__init__()
		self.out_shape = out_shape
		self.kernel_size = kernel_size
		self.strides = strides
		self.std_dev = std_dev

	def call(self, x):
		inp = x

		kernel = K.random_uniform_variable(shape=(self.kernel_size[0], self.kernel_size[1], self.out_shape[-1], int(x.get_shape()[-1])), low=0, high=1)

		deconv = K.conv2d_transpose(x, kernel=kernel, strides=self.strides, output_shape=self.out_shape, padding='same')

		biases = K.zeros(shape=(self.out_shape[-1]))

		deconv = K.reshape(K.bias_add(deconv, biases), deconv.get_shape())
		deconv = LeakyReLU()(deconv)

		g = K.conv2d_transpose(inp, kernel, output_shape=self.out_shape, strides=self.strides, padding='same')
		biases2 = K.zeros(shape=(self.out_shape[-1]))
		g = K.reshape(K.bias_add(g, biases2), deconv.get_shape())

		g = K.sigmoid(g)

		deconv = tf.multiply(deconv, g)

		outputs = [deconv, g]

		output_shapes = self.compute_output_shape(x.shape)
		for output, shape in zip(outputs, output_shapes):
			output._keras_shape = shape

		return [deconv, g]

	def compute_output_shape(self, input_shape):
		return [self.out_shape, self.out_shape]

	def compute_mask(self, input, input_mask=None):
		return 2 * [None]

def data_loader(vars, mode='train'):
	batch_size = vars.SCFEGAN_BATCH_SIZE
	images, _, masks, _, boxes = utils.load_valid_data(vars.SCFEGAN_DATA_INPUT_PATH, batch_size, vars=vars)

	# Removing the void class mask
	masks = np.reshape(masks, (np.shape(masks)[0], vars.INP_SHAPE[0], vars.INP_SHAPE[1], vars.LOGO_NUM_CLASSES))
	reversed_masks = masks[:,:,:,-1]
	masks = masks[:,:,:,:-1]

	detected_masks = [utils.generate_combined_mask(masks[i]) for i in range(np.shape(masks)[0])]

	inputs = []
	all_images = []

	for i, image in enumerate(images):
		input_image = np.array(image)
		random_noise = np.zeros((np.shape(image)[0], np.shape(image)[1], 1))
		random_noise = cv2.randn(random_noise, 0, 255)
		random_noise = np.asarray(random_noise/255, dtype=np.uint8)

		# Normalizing Input Image
		input_image = cv2.resize(input_image, (vars.INP_SHAPE[0], vars.INP_SHAPE[1]))
		input_image = (input_image / 127.5) - 1.

		detected_mask = detected_masks[i]
		reversed_mask = reversed_masks[i]

		img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		sketch = cv2.Canny(img, 100, 100)
		sketch = np.multiply(detected_mask, sketch)

		reversed_mask = np.expand_dims(reversed_mask, axis=-1)
		detected_mask = np.expand_dims(detected_mask, axis=-1)
		sketch = np.expand_dims(sketch, axis=-1)

		input_image = np.multiply(reversed_mask, input_image)
		random_noise = np.multiply(detected_mask, random_noise)

		color = np.multiply(image, detected_mask)

		# input_image = np.expand_dims(input_image, axis=0)
		# random_noise = np.expand_dims(random_noise, axis=0)
		# sketch = np.expand_dims(sketch, axis=0)
		# color = np.expand_dims(color, axis=0)
		# detected_mask = np.expand_dims(detected_mask, axis=0)

		inp = np.concatenate(
			[
				input_image,
				detected_mask,
				sketch,
				color,
				random_noise
			]
		, axis=-1)

		inputs.append(inp)
		all_images.append(image)

	return np.array(inputs), np.array(all_images)
