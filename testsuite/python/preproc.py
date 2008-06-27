"""
Test PreProc
"""

from sg import sg
import util

########################################################################
# kernel computation
########################################################################

def _kernel (indata):
	util.set_and_train_kernel(indata)

	kmatrix=sg('get_kernel_matrix')
	ktrain=max(abs(indata['km_train']-kmatrix).flat)

	sg('init_kernel', 'TEST')
	kmatrix=sg('get_kernel_matrix')
	ktest=max(abs(indata['km_test']-kmatrix).flat)

	return util.check_accuracy(indata['accuracy'], ktrain=ktrain, ktest=ktest)


def _add_preproc (indata):
	pname=util.fix_preproc_name_inconsistency(indata['name'])
	args=util.get_args(indata, 'preproc_arg')

	sg('add_preproc', pname, *args)
	sg('attach_preproc', 'TRAIN')
	sg('attach_preproc', 'TEST')

########################################################################
# public
########################################################################

def test (indata):
	util.set_features(indata)
	_add_preproc(indata)

	return _kernel(indata)

