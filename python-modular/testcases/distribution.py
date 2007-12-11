"""
Test Distribution
"""

from shogun.Distribution import *

import util

def _distribution (indata):
	fun=eval('util.get_feats_'+indata['feature_class'])
	feats=fun(indata)

	if indata['name']=='HMM':
		model=Model()
		distribution=HMM(indata['distribution_N'], indata['distribution_M'],
			model, indata['distribution_pseudo'])
	else:
		fun=eval(indata['name'])
		distribution=fun(feats['train'])

	distribution.train()

	return util.check_accuracy(indata['accuracy'])

########################################################################
# public
########################################################################

def test (indata):
	return _distribution(indata)

