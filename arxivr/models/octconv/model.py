from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam


from .layers import OCTCONV_LAYER

from ..models import BASE

class OCTCONV(BASE):
	def __init__(self, vars, model_name='octconv', input_shape=(None, None, 3)):
		self.inp_shape = input_shape
		super(OCTCONV, self).__init__(vars)

	def compose_model(self):
		# ResNet50 with octconvs

		model = Model(inputs=inp, outputs=out)
		model.compile(loss='', optimizer='')

		return model