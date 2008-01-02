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
from config import C_PREPROC, PREPROC, C_KERNEL

def _compute (name, name_kernel, feats, data, *kargs):
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
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test'])
	}
	outdata.update(fileop.get_outdata(name_kernel, C_KERNEL, kargs))
	return outdata

def _run_string_complex (ftype):
	data=dataop.get_dna()
	name='Sort'+ftype+'String'
	# featop.get_string_complex adds preproc implicitely on Word/Ulong feats
	feats=featop.get_string_complex(ftype, data)
	outdata=_compute(name, 'Comm'+ftype+'String', feats, data,
		False, FULL_NORMALIZATION)
	fileop.write(C_PREPROC, outdata)

def _run_word ():
	name='SortWord'
	data=dataop.get_rand(dattype=ushort)
	feats=featop.get_simple('Word', data)
	feats=featop.add_preproc(name, feats)

	outdata=_compute(name, 'LinearWord', feats, data)
	fileop.write(C_PREPROC, outdata)

def _run_real (name, *args):
	data=dataop.get_rand()
	feats=featop.get_simple('Real', data)
	feats=featop.add_preproc(name, feats, *args)
	
	outdata=_compute(name, 'Gaussian', feats, data, 1.2)
	if args!=():
		outdata.update(fileop.get_args('preproc', PREPROC[name][0], args))
	fileop.write(C_PREPROC, outdata)

def run():
	_run_real('LogPlusOne')
	_run_real('NormOne')
	_run_real('PruneVarSubMean', False)
	_run_real('PruneVarSubMean', True)

	_run_string_complex('Word')
	_run_string_complex('Ulong')
	_run_word()

#	_run_norm_derivative_lem3()
#	_run_pcacut()
