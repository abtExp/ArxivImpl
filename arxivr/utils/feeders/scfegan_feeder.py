from ..feeder import FEEDER
from arxivr.utils.scfegan_utils import data_loader
from arxivr.settings.scfegan_settings import vars

class SCFEGAN_FEEDER(FEEDER):
	def __init__(self, vars, data_loader, mode='train'):
		super(SCFEGAN_FEEDER, self).__init__(vars, data_loader, mode)