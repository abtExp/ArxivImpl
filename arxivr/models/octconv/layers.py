import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer, UpSampling2D, Add, Concatenate, Conv2D, Conv2DTranspose

class OCTCONV_LAYER(Layer):
	def __init__(self,
			filters=16,
			kernel_size=(3, 3),
			strides=(2, 2),
			dilation_rate=(1, 1),
			padding='same',
			alpha_in=0.6,
			alpha_out=0.6,
			**kwargs
		):
		self.filters = filters
		self.kernel_size = kernel_size
		self.strides = strides
		self.padding = padding
		self.alpha_in = alpha_in
		self.alpha_out = alpha_out

		if dilation_rate[0] > 1:
			self.strides = (1, 1)

		self.dilation_rate = dilation_rate
		super(OCTCONV_LAYER, self).__init__(**kwargs)

	def build(self, input_shape):
		op_channels = self.filters
		low_op_channels = int(int(op_channels)*self.alpha_out)
		high_op_channels = op_channels-low_op_channels

		inp_channels = input_shape[-1]
		low_inp_channels = int(int(inp_channels)*self.alpha_in)
		high_inp_channels = int(inp_channels)-low_inp_channels

		self.h_2_l = self.add_weight(
			name='hl',
			shape=self.kernel_size + (high_inp_channels, low_op_channels),
			initializer='he_normal'
		)
		self.h_2_h = self.add_weight(
			name='hh',
			shape=self.kernel_size + (high_inp_channels, high_op_channels),
			initializer='he_normal'
		)
		self.l_2_h = self.add_weight(
			name='lh',
			shape=self.kernel_size + (low_inp_channels, high_op_channels),
			initializer='he_normal'
		)
		self.l_2_l = self.add_weight(
			name='ll',
			shape=self.kernel_size + (low_inp_channels, low_op_channels),
			initializer='he_normal'
		)


		super(OCTCONV_LAYER, self).build(input_shape)

	def call(self, x):
		inp_channels = int(x.shape[-1])
		low_inp_channels = int(int(inp_channels)*self.alpha_in)
		high_inp_channels = int(inp_channels)-low_inp_channels

		high_inp = x[:,:,:, :high_inp_channels]
		low_inp = x[:,:,:, high_inp_channels:]
		low_inp = K.pool2d(low_inp, (2, 2), strides=self.strides, pool_mode='avg')

		out_high_high = K.conv2d(high_inp, self.h_2_h, strides=self.strides, padding=self.padding)
		out_low_high = UpSampling2D((2, 2))(K.conv2d(low_inp, self.l_2_h, strides=self.strides, padding=self.padding))
		out_low_low = K.conv2d(low_inp, self.l_2_l, strides=self.strides, padding=self.padding)
		out_high_low = K.pool2d(high_inp, (2, 2), strides=self.strides, pool_mode='avg')
		out_high_low = K.conv2d(out_high_low, self.h_2_l, strides=self.strides, padding=self.padding)

		out_high = Add()([out_high_high, out_low_high])


		out_low = Add()([out_low_low, out_high_low])

		out_low = UpSampling2D((2, 2))(out_low)


		out_final = K.concatenate([out_high, out_low], axis=-1)

		out_final._keras_shape = self.compute_output_shape(out_final.shape)

		return out_final

	def compute_output_shape(self, inp_shape):
		return inp_shape