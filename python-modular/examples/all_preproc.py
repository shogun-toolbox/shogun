#!/usr/bin/env python
"""
Explicit examples on how to use the different preprocs
"""

from sys import maxint
from numpy import ubyte, ushort, double, int, zeros, sum, floor, array, arange
from numpy.random import randint, rand, seed
from shogun.Kernel import *
from shogun.Features import *
from shogun.PreProc import *

def get_dna ():
	acgt=array(['A', 'C', 'G','T'])
	len_acgt=len(acgt)
	rand_train=[]
	rand_test=[]

	for i in xrange(11):
		str1=[]
		str2=[]
		for j in range(60):
			str1.append(acgt[floor(len_acgt*rand())])
			str2.append(acgt[floor(len_acgt*rand())])
		rand_train.append(''.join(str1))
	rand_test.append(''.join(str2))
	
	for i in xrange(6):
		str1=[]
		for j in range(60):
			str1.append(acgt[floor(len_acgt*rand())])
	rand_test.append(''.join(str1))

	return {'train': rand_train, 'test': rand_test}

###########################################################################
# real features
###########################################################################

def log_plus_one ():
	print 'LogPlusOne'

	num_feats=11
	data=rand(num_feats, 11)
	feats_train=RealFeatures(data)
	data=rand(num_feats, 17)
	feats_test=RealFeatures(data)

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

	num_feats=11
	data=rand(num_feats, 11)
	feats_train=RealFeatures(data)
	data=rand(num_feats, 17)
	feats_test=RealFeatures(data)

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

	num_feats=11
	data=rand(num_feats, 11)
	feats_train=RealFeatures(data)
	data=rand(num_feats, 17)
	feats_test=RealFeatures(data)

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
# word features
###########################################################################

def sort_word ():
	print 'LinearWord'

	maxval=2**16-1
	num_feats=11
	data=randint(0, maxval, (num_feats, 11)).astype(ushort)
	feats_train=WordFeatures(data)
	data=randint(0, maxval, (num_feats, 17)).astype(ushort)
	feats_test=WordFeatures(data)

	preproc=SortWord()
	preproc.init(feats_train)
	feats_train.add_preproc(preproc)
	feats_train.apply_preproc()
	feats_test.add_preproc(preproc)
	feats_test.apply_preproc()

	do_rescale=True
	scale=1.4

	kernel=LinearWordKernel(feats_train, feats_train, do_rescale, scale)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

###########################################################################
# complex string features
###########################################################################

def sort_word_string ():
	print 'CommWordString'

	data=get_dna()
	order=3
	gap=0
	reverse=False

	charfeat=StringCharFeatures(DNA)
	charfeat.set_string_features(data['train'])
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=SortWordString()
	preproc.init(feats_train)
	feats_train.add_preproc(preproc)
	feats_train.apply_preproc()

	charfeat=StringCharFeatures(DNA)
	charfeat.set_string_features(data['test'])
	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats_test.add_preproc(preproc)
	feats_test.apply_preproc()

	use_sign=False
	normalization=FULL_NORMALIZATION

	kernel=CommWordStringKernel(
		feats_train, feats_train, use_sign, normalization)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

def sort_ulong_string ():
	print 'CommUlongString'

	data=get_dna()
	order=3
	gap=0
	reverse=False

	charfeat=StringCharFeatures(DNA)
	charfeat.set_string_features(data['train'])
	feats_train=StringUlongFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)

	charfeat=StringCharFeatures(DNA)
	charfeat.set_string_features(data['test'])
	feats_test=StringUlongFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)

	preproc=SortUlongString()
	preproc.init(feats_train)
	feats_train.add_preproc(preproc)
	feats_train.apply_preproc()
	feats_test.add_preproc(preproc)
	feats_test.apply_preproc()

	use_sign=False
	normalization=FULL_NORMALIZATION

	kernel=CommUlongStringKernel(
		feats_train, feats_train, use_sign, normalization)

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

	sort_word()

	sort_word_string()
	sort_ulong_string()
