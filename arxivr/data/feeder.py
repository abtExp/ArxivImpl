import os
import cv2
import numpy as np

'''
Data Feeder Is Required For Custom Loading Of Files
'''

class DATA_FEEDER():
	def __init__(self, config, feed_function):
		self.config = config
		self.feed_function = feed_function

	def get_batch(self):
		return self.feed_function()

	def shuffle(self):
		return

	def augment(self):
		return

	def split(self):
		return

	def preprocess(self):
		return