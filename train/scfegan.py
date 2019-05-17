import sys
sys.path.insert(0, './settings')
sys.path.insert(0, './1902.06838')

import vars
from model import SCFEGAN

model = SCFEGAN(vars)
model.summary()
model.plot()