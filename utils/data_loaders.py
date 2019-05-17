import sys
sys.path.insert(0, './utils')

from scfegan_utils import data_loader
from enet_utils import data_loader

from keras.utils import Sequence

class DATA_LOADER(Sequence):
	def __init__(self, model='enet', vars={}, mode='train'):
		self.vars = vars
		self.mode = mode
		self.loader = self.vars[model.upper()]['loader']

	def __getitem__(self, idx):
		return x, y

	def __data_generation(self, k):
		train_x, train_y = self.loader(self.vars, self.mode)
		return train_x, train_y