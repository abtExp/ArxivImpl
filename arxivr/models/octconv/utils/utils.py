from ..data import DATA_LOADER, DATA_FEEDER
from ..settings import CONFIG


config = CONFIG('./settings/config.json')

class FEEDER(DATA_FEEDER):
	def __init__(self, config):
		super(FEEDER, self).__init__(config)


loader = DATA_LOADER()