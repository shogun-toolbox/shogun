#!/usr/bin/env python
"""
Explicit examples on how to use the different preprocs
"""

from sys import maxint
from numpy import char, ushort, double, int, zeros, sum, floor, array, arange
from numpy.random import randint, rand, seed
from shogun.Kernel import *
from shogun.Features import *
from shogun.PreProc import *

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

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	preproc=LogPlusOne()
	preproc.init(feats_train)
	feats_train.add_preproc(preproc)
	feats_train.apply_preproc()
	feats_test.add_preproc(preproc)
	feats_test.apply_preproc()

	width=1.4
	size_cache=10
	
	kernel=Chi2Kernel(feats_train, feats_train, width, size_cache)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

def norm_one ():
	print 'NormOne'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	preproc=NormOne()
	preproc.init(feats_train)
	feats_train.add_preproc(preproc)
	feats_train.apply_preproc()
	feats_test.add_preproc(preproc)
	feats_test.apply_preproc()

	width=1.4
	size_cache=10
	
	kernel=Chi2Kernel(feats_train, feats_train, width, size_cache)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

def prune_var_sub_mean ():
	print 'PruneVarSubMean'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	preproc=PruneVarSubMean()
	preproc.init(feats_train)
	feats_train.add_preproc(preproc)
	feats_train.apply_preproc()
	feats_test.add_preproc(preproc)
	feats_test.apply_preproc()

	width=1.4
	size_cache=10
	
	kernel=Chi2Kernel(feats_train, feats_train, width, size_cache)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

###########################################################################
# complex string features
###########################################################################

def sort_word_string ():
	print 'CommWordString'

	order=3
	gap=0
	reverse=False

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_train_dna)
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=SortWordString()
	preproc.init(feats_train)
	feats_train.add_preproc(preproc)
	feats_train.apply_preproc()

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_test_dna)
	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats_test.add_preproc(preproc)
	feats_test.apply_preproc()

	use_sign=False

	kernel=CommWordStringKernel(feats_train, feats_train, use_sign)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

def sort_ulong_string ():
	print 'CommUlongString'

	order=3
	gap=0
	reverse=False

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_train_dna)
	feats_train=StringUlongFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_test_dna)
	feats_test=StringUlongFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)

	preproc=SortUlongString()
	preproc.init(feats_train)
	feats_train.add_preproc(preproc)
	feats_train.apply_preproc()
	feats_test.add_preproc(preproc)
	feats_test.apply_preproc()

	use_sign=False

	kernel=CommUlongStringKernel(feats_train, feats_train, use_sign)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

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
