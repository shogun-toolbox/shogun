"""Generator for Preprocessors"""

import numpy
import shogun.Library as library
from shogun.Kernel import GaussianKernel, CommWordStringKernel, \
	CommUlongStringKernel, LinearWordKernel

import fileop
import featop
import dataop
import config

def _compute (name, name_kernel, feats, data, *kargs):
	"""Perform computations on kernel using preprocessors.

	@param name Name of the preprocessor
	@param name_kernel Name of the kernel
	@param feats Features of the kernel
	@param data Train and test data
	@param *kargs Variable kernel argument list
	@return Dict of testcase data ready to be written to file
	"""

	fun=eval(name_kernel+'Kernel')
	kernel=fun(feats['train'], feats['train'], *kargs)
	km_train=kernel.get_kernel_matrix()
	kernel.init(feats['train'], feats['test'])
	km_test=kernel.get_kernel_matrix()

	outdata={
		'name':name,
		'name_kernel':name_kernel,
		'km_train':km_train,
		'km_test':km_test,
		'data_train':numpy.matrix(data['train']),
		'data_test':numpy.matrix(data['test'])
	}
	outdata.update(fileop.get_outdata(name_kernel, config.C_KERNEL, kargs))
	return outdata

def _run_string_complex (ftype):
	"""Run preprocessor applied on complex StringFeatures.

	@param ftype Feature type, like Word
	"""

	data=dataop.get_dna()
	name='Sort'+ftype+'String'
	# featop.get_string_complex adds preproc implicitely on Word/Ulong feats
	feats=featop.get_string_complex(ftype, data)
	outdata=_compute(name, 'Comm'+ftype+'String', feats, data,
		False, library.FULL_NORMALIZATION)
	fileop.write(config.C_PREPROC, outdata)

def _run_word ():
	"""Run preprocessor applied on WordFeatures."""

	name='SortWord'
	data=dataop.get_rand(dattype=numpy.ushort)
	feats=featop.get_simple('Word', data)
	feats=featop.add_preproc(name, feats)

	outdata=_compute(name, 'LinearWord', feats, data)
	fileop.write(config.C_PREPROC, outdata)

def _run_real (name, *args):
	"""Run preprocessor applied on RealFeatures.

	@param name Name of the preprocessor
	@param args Variable argument list for the preprocessor
	"""

	data=dataop.get_rand()
	feats=featop.get_simple('Real', data)
	feats=featop.add_preproc(name, feats, *args)
	
	outdata=_compute(name, 'Gaussian', feats, data, 1.2)
	if args!=():
		outdata.update(fileop.get_args('preproc', config.PREPROC[name][0], args))
	fileop.write(config.C_PREPROC, outdata)

def run():
	"""Run generator for all preprocessors."""

	_run_real('LogPlusOne')
	_run_real('NormOne')
	_run_real('PruneVarSubMean', False)
	_run_real('PruneVarSubMean', True)

	_run_string_complex('Word')
	_run_string_complex('Ulong')
	_run_word()

#	_run_norm_derivative_lem3()
#	_run_pcacut()
