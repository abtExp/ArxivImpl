from .layers import OCTCONV_LAYER

from ..base import BASE

class OCTCONV(BASE):
	def __init__(self, config):
		self.config = config
		super(OCTCONV, self).__init__(vars)

	def compose_model(self):
		# ResNet50 with octconvs

		model = Model(inputs=inp, outputs=out)
		model.compile(loss='', optimizer='')

		return model