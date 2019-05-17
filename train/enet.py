import sys
sys.path.insert(0, './settings')
sys.path.insert(0, './1606.02147')

import vars
from model import ENET

model = ENET(vars)
model.summary()