"""
Test Preprocessor
"""

from modshogun import *

import util


########################################################################
# kernel computation
########################################################################

def _evaluate (indata):
	prefix='kernel_'
	feats=util.get_features(indata, prefix)
	kfun=eval(indata[prefix+'name']+'Kernel')
	kargs=util.get_args(indata, prefix)

	prefix='preprocessor_'
	pargs=util.get_args(indata, prefix)
	feats=util.add_preprocessor(indata[prefix+'name'], feats, *pargs)

	prefix='kernel_'
	kernel=kfun(feats['train'], feats['train'], *kargs)
	km_train=max(abs(
		indata[prefix+'matrix_train']-kernel.get_kernel_matrix()).flat)
	kernel.init(feats['train'], feats['test'])
	km_test=max(abs(
		indata[prefix+'matrix_test']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy(
		indata[prefix+'accuracy'], km_train=km_train, km_test=km_test)


########################################################################
# public
########################################################################

def test (indata):
	return _evaluate(indata)

