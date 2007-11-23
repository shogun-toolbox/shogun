from shogun.Distance import *

import fileops
import dataops
import featops
from kernels import compute

def run ():
	data=dataops.get_rand()
	feats=featops.get_simple('Real', data)
	distance=CanberraMetric()
	distance.init(feats['train'], feats['train'])

	fileops.write(compute('Distance', feats, data, 1.5, distance))


