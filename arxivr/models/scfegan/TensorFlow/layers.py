from tensorflow.keras.layers import Layer, Conv2D, Conv2DTranspose, \
									Reshape, Multiply, LeakyReLU, \
									Activation, multiply, BatchNormalization

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
	def __init__(self, out_shape, kernel_size=(5, 5), strides=(2, 2), std_dev=0.02):
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
				trainable=False
			)

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


def GatedConv2D(x, filters, kernel_size, strides, dilation=1, activation='leaky_relu', use_lrn=True):
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


# From https://medium.com/@FloydHsiu0618/spectral-normalization-implementation-of-tensorflow-2-0-keras-api-d9060d26de77
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

class SpectralNormalization(layers.Wrapper):
	"""
	Attributes:
	layer: tensorflow keras layers (with kernel attribute)
	"""

	def __init__(self, layer, **kwargs):
		super(SpectralNormalization, self).__init__(layer, **kwargs)

	def build(self, input_shape):
		"""Build `Layer`"""

		if not self.layer.built:
			self.layer.build(input_shape)

		if not hasattr(self.layer, 'kernel'):
			raise ValueError(
				'`SpectralNormalization` must wrap a layer that'
				' contains a `kernel` for weights')

		self.w = self.layer.kernel
		self.w_shape = self.w.shape.as_list()
		self.u = self.add_variable(
			shape=tuple([1, self.w_shape[-1]]),
			initializer=initializers.TruncatedNormal(stddev=0.02),
			name='sn_u',
			trainable=False,
			dtype=dtypes.float32)

		super(SpectralNormalization, self).build()

	@def_function.function
	def call(self, inputs, training=None):
		"""Call `Layer`"""
		if training is None:
			training = K.learning_phase()

		if training==True:
			# Recompute weights for each forward pass
			self._compute_weights()

		output = self.layer(inputs)
		return output

	def _compute_weights(self):
		"""Generate normalized weights.
		This method will update the value of self.layer.kernel with the
		normalized value, so that the layer is ready for call().
		"""
		w_reshaped = array_ops.reshape(self.w, [-1, self.w_shape[-1]])
		eps = 1e-12
		_u = array_ops.identity(self.u)
		_v = math_ops.matmul(_u, array_ops.transpose(w_reshaped))
		_v = _v / math_ops.maximum(math_ops.reduce_sum(_v**2)**0.5, eps)
		_u = math_ops.matmul(_v, w_reshaped)
		_u = _u / math_ops.maximum(math_ops.reduce_sum(_u**2)**0.5, eps)

		self.u.assign(_u)
		sigma = math_ops.matmul(math_ops.matmul(_v, w_reshaped), array_ops.transpose(_u))

		self.layer.kernel.assign(self.w / sigma)

	def compute_output_shape(self, input_shape):
		return tensor_shape.TensorShape(
			self.layer.compute_output_shape(input_shape).as_list())