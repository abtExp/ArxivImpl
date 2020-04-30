from keras.layers import Layer, Dense, BatchNormalization, ReLU, Add, Multiply
import keras.backend as K
import numpy as np

class GROUPED_LINEAR(Layer):
	def __init__(self, groups, output_units):
		self.groups = groups
		self.output_units = output_units

	def call(self, x):
		self.layers = self.groups
		return

	def compute_output_shape(self, input_shape):
		return None

class GATEDCONV(Layer):
	def __init__(self, filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1), pool_size=(None, None), groups=None):
		self.filters = filters
		self.kernel_size = kernel_size
		self.strides = strides
		self.padding = padding
		self.dilation_rate = dilation_rate

		# Setting the group size for grouped linear layer in channel interacting module
		if not groups:
			groups = 1/16

		self.groups = groups

		# Setting the poolsize for the initial resizing of the input
		if not pool_size[0]:
			pool_size = kernel_size

		self.pool_size = pool_size

		super(GATEDCONV, self).__init__()

	def build(self, input_shape):
		# Add the shape according to the dilation
		self.kernel = self.add_weights(
			name='kernel',
			shape=(None, self.filters, *self.kernel_size),
			initializer='he_uniform'
		)

		super(GATEDCONV, self).build(input_shape)

	def call(self, x):
		inp_channels = x.shape[-1]

		if self.kernel_size[0] != 1:
			inp = x
			print(inp.shape)
			# (-> (h, w, c))
			x = K.pool2d(x, pool_size=self.pool_size, pool_mode='avg')
			print(x.shape)
			# Context encoding (-> (h', w', c))
			enc = K.reshape(x, (x.shape[1]*x.shape[2], x.shape[-1]))
			print(enc.shape)
			# (-> (h'xw', c))
			enc = K.transpose(enc)
			print(enc.shape)
			# (-> (c, h'xw'))
			enc = Dense(units=int(np.product(self.kernel_size)/2))(enc)
			print(enc.shape)
			# (-> (c, d) : d = (kernel_heightxkernel_width)/2)
			enc = BatchNormalization()(enc)
			enc = ReLU()(enc)

			# Channel interaction
					# Does depthwise-separable operation in groups rather than per channel
			# (-> (c, d))
			channel = GROUPED_LINEAR(self.groups, self.filters)(enc)
			# (-> (o, d))
			channel = BatchNormalization()(channel)
			channel = ReLU()(channel)

			# Gated Decoding
			# (-> (o, d))
			decoding_layer = Dense(units=np.product(self.kernel_size))
			input_channels = decoding_layer(enc)
			output_channels = decoding_layer(channel)
			decoded = K.sigmoid(Add()([input_channels, output_channels]))
			decoded = K.reshape(decoded, (self.filters, inp_channels, *self.kernel_size))
			# (-> (o, c, k1, k2))

			new_kernel = Multiply()([self.kernel, decoded])
			self.kernel = new_kernel

		return K.conv2d(inp, self.kernel, self.strides, self.padding, dilation_rate=self.dilation_rate)

	def compute_output_shape(self, input_shape):
		return