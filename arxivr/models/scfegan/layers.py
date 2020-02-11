from keras.layers import Layer, Conv2D, Conv2DTranspose, Reshape, Multiply, LeakyReLU, Activation, multiply, BatchNormalization
from keras.initializers import RandomNormal
import keras.backend as K
import tensorflow as tf

import numpy as np

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

		return deconv, g

	def compute_output_shape(self, input_shape):
		return [self.out_shape, self.out_shape]

	def compute_mask(self, input, input_mask=None):
		return [None]