from shogun.Distribution import *

import util

def _distribution (input):
	fun=eval('util.get_feats_'+input['feature_class'])
	feats=fun(input)

	if input['name']=='HMM':
		model=Model()
		distribution=HMM(input['distribution_N'], input['distribution_M'],
			model, input['distribution_pseudo'])
	else:
		fun=eval(input['name'])
		distribution=fun(feats['train'])

	distribution.train()

	return util.check_accuracy(input['accuracy'])

########################################################################
# public
########################################################################

def test (input):
	return _distribution(input)

