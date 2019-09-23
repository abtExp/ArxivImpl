from keras.layers import Layer, Conv2D, Conv2DTranspose, Reshape, Multiply, LeakyReLU, Activation, multiply, BatchNormalization
from keras.initializers import RandomNormal
import keras.backend as K
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import cv2

import os

import face_recognition

import math

from os import listdir


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


def create_sketch():
	return sketch

# TODO : Create Hair Mask
def get_hair_mask():
	return np.zeros((256, 256), dtype=np.int32)

# Based On The Algorithm Mentioned In The Paper (Algorithm 1)
def create_mask(img, max_draws=10, max_len=50, max_angle=60, max_lines=10, shape=(256, 256)):
	mask_panel = np.zeros(shape)

	num_lines = np.random.randint(0, max_draws)

	bbox = face_recognition.face_locations(img)[0]

	if len(bbox) > 0:
		mask = np.zeros((bbox[1]-bbox[3], bbox[2]-bbox[0]))

		for i in range(0, num_lines):
			start_x = np.random.randint(0, bbox[2]-bbox[0])
			start_y = np.random.randint(0, bbox[1]-bbox[3])
			start_angle = np.random.randint(0, 360)
			num_vertices = np.random.randint(0, max_lines)

			for j in range(0, num_vertices):
				angle_change = np.random.randint(-max_angle, max_angle)
				if j%2 == 0:
					angle = start_angle + angle_change
				else:
					angle = start_angle + angle_change + 180

				length = np.random.randint(0, max_len)

				end_x = start_x+int(length * math.cos(math.radians(angle)))
				end_y = start_y+int(length * math.sin(math.radians(angle)))

				mask = cv2.line(mask, (start_x, start_y), (end_x, end_y), (255, 255, 255), 10)

				start_x = end_x
				start_y = end_y

		mask = np.array(mask, dtype='int32')
		mask_panel[bbox[3]:bbox[1], bbox[0]:bbox[2]] = mask[:,:]

		# if np.random.randint(0, 10) > 5:
		# 	hair_mask = get_hair_mask()
		# 	mask += hair_mask

		return mask_panel

	else:
		return