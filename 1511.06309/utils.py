# Utility Methods

from keras.layers import Layer

class HuberLossLayer(Layer):
	def __init__(self):
		super(HuberLossLayer, self).__init__()
	
	def call(self, x):
		if abs(x) <= delta:
			return 0.5*(x^2)
		else:
			return delta*(abs(x) - 0.5*delta)
