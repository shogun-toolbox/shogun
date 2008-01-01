"""
Generator for Preprocessors
"""

from numpy import matrix, ushort
from shogun.Library import FULL_NORMALIZATION
from shogun.Kernel import *
from shogun.PreProc import *

import fileop
import featop
import dataop
from config import C_PREPROC, C_KERNEL

def _compute (name, name_kernel, feats, data, *args):
	fun=eval(name_kernel+'Kernel')
	kernel=fun(feats['train'], feats['train'], *args)
	km_train=kernel.get_kernel_matrix()
	kernel.init(feats['train'], feats['test'])
	km_test=kernel.get_kernel_matrix()

	outdata={
		'name':name,
		'name_kernel':name_kernel,
		'km_train':km_train,
		'km_test':km_test,
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test'])
	}
	outdata.update(fileop.get_outdata(name_kernel, C_KERNEL, args))
	fileop.write(C_PREPROC, outdata)

def _run_string_complex ():
	data=dataop.get_dna()

	name='SortWordString'
	# featop.get_string_complex does SortWordString implicitely on Word feats
	feats=featop.get_string_complex('Word', data)
	_compute(name, 'CommWordString', feats, data, False, FULL_NORMALIZATION)

	name='SortUlongString'
	feats=featop.get_string_complex('Ulong', data)
	feats=featop.add_preproc(name, feats)
	_compute(name, 'CommUlongString', feats, data, False, FULL_NORMALIZATION)

def _run_word ():
	name='SortWord'
	data=dataop.get_rand(dattype=ushort)
	feats=featop.get_simple('Word', data)
	feats=featop.add_preproc(name, feats)
	_compute(name, 'LinearWord', feats, data)

def _run_real (name):
	data=dataop.get_rand()
	feats=featop.get_simple('Real', data)
	feats=featop.add_preproc(name, feats)
	_compute(name, 'Gaussian', feats, data, 1.2)

def run():
	_run_string_complex()
	_run_real('LogPlusOne')
	_run_real('NormOne')
	_run_real('PruneVarSubMean')
	_run_word()
#	_run_norm_derivative_lem3()
#	_run_pcacut()
