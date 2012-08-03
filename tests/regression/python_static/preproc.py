"""
Test PreProc
"""

from sg import sg
import util


def _evaluate (indata, prefix):
	util.set_and_train_kernel(indata)

	kmatrix=sg('get_kernel_matrix', 'TRAIN')
	km_train=max(abs(indata['kernel_matrix_train']-kmatrix).flat)

	kmatrix=sg('get_kernel_matrix', 'TEST')
	km_test=max(abs(indata['kernel_matrix_test']-kmatrix).flat)

	return util.check_accuracy(
		indata[prefix+'accuracy'], km_train=km_train, km_test=km_test)


def _set_preproc (indata, prefix):
	pname=util.fix_preproc_name_inconsistency(indata[prefix+'name'])
	args=util.get_args(indata, prefix)

	sg('add_preproc', pname, *args)
	sg('attach_preproc', 'TRAIN')
	sg('attach_preproc', 'TEST')


########################################################################
# public
########################################################################

def test (indata):
	prefix='kernel_'
	try:
		util.set_features(indata, prefix)
	except NotImplementedError, e:
		print e
		return True

	_set_preproc(indata, 'preproc_')

	return _evaluate(indata, prefix)

