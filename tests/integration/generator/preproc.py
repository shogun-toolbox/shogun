"""Generator for Preprocessors"""

import shogun.Library as library
from shogun.Kernel import GaussianKernel, CommWordStringKernel, \
	CommUlongStringKernel

import fileop
import featop
import dataop
import category


def _compute (feats, params):
	"""Perform computations on kernel using preprocessors.

	@param name name of the kernel
	@param feats features of the kernel
	@return dict of testcase data ready to be written to file
	"""

	output=fileop.get_output(category.KERNEL, params)

	fun=eval(params['name']+'Kernel')
	if params.has_key('args'):
		kernel=fun(feats['train'], feats['train'], *params['args']['val'])
	else:
		kernel=fun(feats['train'], feats['train'])

	output['kernel_matrix_train']=kernel.get_kernel_matrix()
	kernel.init(feats['train'], feats['test'])
	output['kernel_matrix_test']=kernel.get_kernel_matrix()

	return output


def _run_string_complex (ftype):
	"""Run preprocessor applied on complex StringFeatures.

	@param ftype Feature type, like Word
	"""

	params={
		'name': 'Comm'+ftype+'String',
		'accuracy': 1e-9,
		'feature_class': 'string_complex',
		'feature_type': ftype,
		'data': dataop.get_dna()
	}
	feats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'])
	# string_complex gets preproc added implicitely on Word/Ulong feats
	output=_compute(feats, params)

	params={
		'name': 'Sort'+ftype+'String'
	}
	output.update(fileop.get_output(category.PREPROC, params))

	fileop.write(category.PREPROC, output)


def _run_real (name, args=None):
	"""Run preprocessor applied on RealFeatures.

	@param name name of the preprocessor
	@param args argument list (in a dict) for the preprocessor
	"""

	params={
		'name': 'Gaussian',
		'accuracy': 1e-8,
		'data': dataop.get_rand(),
		'feature_class': 'simple',
		'feature_type': 'Real',
		'args': {'key': ('width',), 'val': (1.2,)}
	}
	feats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'])
	if args:
		feats=featop.add_preproc(name, feats, *args['val'])
	else:
		feats=featop.add_preproc(name, feats)

	output=_compute(feats, params)

	params={ 'name': name }
	if args:
		params['args']=args

	output.update(fileop.get_output(category.PREPROC, params))

	fileop.write(category.PREPROC, output)


def run():
	"""Run generator for all preprocessors."""

	_run_real('LogPlusOne')
	_run_real('NormOne')
	_run_real('PruneVarSubMean', {'key': ('divide',), 'val': (False,)})
	_run_real('PruneVarSubMean', {'key': ('divide',), 'val': (True,)})

	_run_string_complex('Word')
	_run_string_complex('Ulong')

#	_run_norm_derivative_lem3()
#	_run_pcacut()
