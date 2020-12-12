from settings.config import CONFIG
from TensorFlow.model import SCFEGAN

config = CONFIG('./settings/config.json')
model = SCFEGAN(config)
# model.train()
model.summary()
model.plot()