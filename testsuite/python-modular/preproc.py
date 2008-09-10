"""
Test PreProc
"""

from shogun.Kernel import *

import util

########################################################################
# kernel computation
########################################################################

def _kernel (indata, feats):
#	print indata['kernel_name']
	fun=eval(indata['kernel_name']+'Kernel')
	args=util.get_args(indata, 'kernel_arg')

#	print args
	kernel=fun(feats['train'], feats['train'], *args)
	ktrain=max(abs(indata['km_train']-kernel.get_kernel_matrix()).flat)
#	a=kernel.get_kernel_matrix()
#	b=indata['km_train']
#	print a[0][0]
#	print b[0][0]
#	print a[0][0]-b[0][0]
#	import pdb
#	pdb.set_trace()

	kernel.init(feats['train'], feats['test'])
	ktest=max(abs(indata['km_test']-kernel.get_kernel_matrix()).flat)

	return util.check_accuracy(indata['accuracy'], ktrain=ktrain, ktest=ktest)

########################################################################
# public
########################################################################

def test (indata):
	fun=eval('util.get_feats_'+indata['feature_class'])
	feats=fun(indata)
	args=util.get_args(indata, 'preproc_arg')
	feats=util.add_preproc(indata['name'], feats, *args)

	return _kernel(indata, feats)

