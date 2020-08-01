from tensorflow.keras.layers import Layer, Conv2D, Conv2DTranspose, Reshape, Multiply, LeakyReLU, Activation, multiply, BatchNormalization
from tensorflow.keras.initializers import RandomNormal
import tensorflow.keras.backend as K
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


def kernel_spectral_norm(kernel, iteration=1, name='kernel_sn'):
	# spectral_norm
	def l2_norm(input_x, epsilon=1e-12):
		input_x_norm = input_x / (tf.reduce_sum(input_x**2)**0.5 + epsilon)
		return input_x_norm
	with tf.variable_scope(name) as scope:
		w_shape = kernel.get_shape().as_list()
		w_mat = tf.reshape(kernel, [-1, w_shape[-1]])
		u = tf.get_variable(
			'u', shape=[1, w_shape[-1]],
			initializer=tf.truncated_normal_initializer(),
			trainable=False)

		def power_iteration(u, ite):
			v_ = tf.matmul(u, tf.transpose(w_mat))
			v_hat = l2_norm(v_)
			u_ = tf.matmul(v_hat, w_mat)
			u_hat = l2_norm(u_)
			return u_hat, v_hat, ite+1

		u_hat, v_hat,_ = power_iteration(u, iteration)
		sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
		w_mat = w_mat / sigma
		with tf.control_dependencies([u.assign(u_hat)]):
			w_norm = tf.reshape(w_mat, w_shape)
		return w_norm

class SpectralNormedConv2D(tf.keras.layers.Conv2D):
	def build(self, input_shape):
		super(SpectralNormedConv2D, self).build(input_shape)
		self.kernel = kernel_spectral_norm(self.kernel)