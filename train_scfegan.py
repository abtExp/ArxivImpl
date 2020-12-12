from arxivr.models import SCFEGAN
from arxivr.models.scfegan.settings.config import CONFIG

config = CONFIG('D:/exp_labs/arxivr/arxivr/models/scfegan/settings/config.json')

model = SCFEGAN(config)