from settings import vars
from models import SCFEGAN

model = SCFEGAN(vars)

model.train()