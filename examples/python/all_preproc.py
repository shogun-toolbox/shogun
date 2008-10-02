#!/usr/bin/env python
"""
Explicit examples on how to use the different preprocs
"""

from sys import maxint
from numpy import char, ushort, double, int, zeros, sum, floor, array, arange
from numpy.random import randint, rand, seed
from sg import sg

from tools.load import LoadMatrix
lm=LoadMatrix()
fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
fm_train_word=ushort(lm.load_numbers('../data/fm_train_real.dat'))
fm_test_word=ushort(lm.load_numbers('../data/fm_test_real.dat'))
fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')

###########################################################################
# real features
###########################################################################

def log_plus_one ():
	print 'LogPlusOne'

	width=1.4
	size_cache=10

	sg('add_preproc', 'LOGPLUSONE')
	sg('set_kernel', 'CHI2', 'REAL', size_cache, width)

	sg('set_features', 'TRAIN', fm_train_real)
	sg('attach_preproc', 'TRAIN')
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('set_features', 'TEST', fm_test_real)
	sg('attach_preproc', 'TEST')
	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def norm_one ():
	print 'NormOne'

	width=1.4
	size_cache=10

	sg('add_preproc', 'NORMONE')
	sg('set_kernel', 'CHI2', 'REAL', size_cache, width)

	sg('set_features', 'TRAIN', fm_train_real)
	sg('attach_preproc', 'TRAIN')
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('set_features', 'TEST', fm_test_real)
	sg('attach_preproc', 'TEST')
	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def prune_var_sub_mean ():
	print 'PruneVarSubMean'

	width=1.4
	size_cache=10
	divide_by_std=True

	sg('add_preproc', 'PRUNEVARSUBMEAN', divide_by_std)
	sg('set_kernel', 'CHI2', 'REAL', size_cache, width)

	sg('set_features', 'TRAIN', fm_train_real)
	sg('attach_preproc', 'TRAIN')
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('set_features', 'TEST', fm_test_real)
	sg('attach_preproc', 'TEST')
	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

###########################################################################
# complex string features
###########################################################################

def sort_word_string ():
	print 'CommWordString'

	size_cache=10
	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true
	use_sign=False
	normalization='FULL'

	sg('add_preproc', 'SORTWORDSTRING')
	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')

	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TEST')

	sg('set_kernel', 'COMMSTRING', 'WORD', size_cache, use_sign, normalization)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def sort_ulong_string ():
	print 'CommUlongString'

	size_cache=10
	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true
	use_sign=False
	normalization='FULL'

	sg('add_preproc', 'SORTULONGSTRING')
	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')

	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse)
	sg('attach_preproc', 'TEST')

	sg('set_kernel', 'COMMSTRING', 'ULONG', size_cache, use_sign, normalization)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	seed(42)

	log_plus_one()
	norm_one()
	prune_var_sub_mean()

	sort_word_string()
	sort_ulong_string()
