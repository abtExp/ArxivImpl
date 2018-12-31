# Utility Methods

from keras.layers import Layer
import numpy as np

class HuberLossLayer(Layer):
	def __init__(self):
		super(HuberLossLayer, self).__init__()
	
	def call(self, x):
		delta = 1e-3
		if abs(x) <= delta:
			return 0.5*(x^2)
		else:
			return delta*(abs(x) - 0.5*delta)


class GridGen():
	def __init__(self, height, width):
		self.height = height
		self.width = width

		