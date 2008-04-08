#!/usr/bin/env python
"""
Explicit examples on how to use the different preprocs
"""

from sys import maxint
from numpy import ubyte, ushort, double, int, zeros, sum, floor, array, arange
from numpy.random import randint, rand, seed
from sg import sg

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
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)
	width=1.4
	size_cache=10

	sg('send_command', 'add_preproc LOGPLUSONE')
	sg('send_command', 'set_kernel CHI2 REAL %d %f' % (size_cache, width))

	sg('set_features', 'TRAIN', traindata)
	sg('send_command', 'attach_preproc TRAIN')
	sg('send_command', 'init_kernel TRAIN')
	km=sg('get_kernel_matrix')

	sg('set_features', 'TEST', testdata)
	sg('send_command', 'attach_preproc TEST')
	sg('send_command', 'init_kernel TEST')
	km=sg('get_kernel_matrix')

def norm_one ():
	print 'NormOne'

	num_feats=11
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)
	width=1.4
	size_cache=10

	sg('send_command', 'add_preproc NORMONE')
	sg('send_command', 'set_kernel CHI2 REAL %d %f' % (size_cache, width))

	sg('set_features', 'TRAIN', traindata)
	sg('send_command', 'attach_preproc TRAIN')
	sg('send_command', 'init_kernel TRAIN')
	km=sg('get_kernel_matrix')

	sg('set_features', 'TEST', testdata)
	sg('send_command', 'attach_preproc TEST')
	sg('send_command', 'init_kernel TEST')
	km=sg('get_kernel_matrix')

def prune_var_sub_mean ():
	print 'PruneVarSubMean'

	num_feats=11
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)
	width=1.4
	size_cache=10
	divide_by_std=1

	sg('send_command', 'add_preproc PRUNEVARSUBMEAN %d' % divide_by_std)
	sg('send_command', 'set_kernel CHI2 REAL %d %f' % (size_cache, width))

	sg('set_features', 'TRAIN', traindata)
	sg('send_command', 'attach_preproc TRAIN')
	sg('send_command', 'init_kernel TRAIN')
	km=sg('get_kernel_matrix')

	sg('set_features', 'TEST', testdata)
	sg('send_command', 'attach_preproc TEST')
	sg('send_command', 'init_kernel TEST')
	km=sg('get_kernel_matrix')

###########################################################################
# word features
###########################################################################

def sort_word ():
	print 'LinearWord'

	maxval=2**16-1
	num_feats=11
	traindata=randint(0, maxval, (num_feats, 11)).astype(ushort)
	testdata=randint(0, maxval, (num_feats, 17)).astype(ushort)
	size_cache=10
	scale=1.4

	sg('send_command', 'add_preproc SORTWORD')
	sg('send_command', 'set_kernel LINEAR WORD %d %f' % (size_cache, scale))

	sg('set_features', 'TRAIN', traindata)
	sg('send_command', 'attach_preproc TRAIN')
	sg('send_command', 'init_kernel TRAIN')
	km=sg('get_kernel_matrix')

	sg('set_features', 'TEST', testdata)
	sg('send_command', 'attach_preproc TEST')
	sg('send_command', 'init_kernel TEST')
	km=sg('get_kernel_matrix')

###########################################################################
# complex string features
###########################################################################

def sort_word_string ():
	print 'CommWordString'

	data=get_dna()
	size_cache=10
	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true
	use_sign=False
	normalization='FULL'

	sg('send_command', 'add_preproc SORTWORDSTRING')
	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('send_command', 'convert TRAIN STRING CHAR STRING WORD %d %d %d %c' % (order, order-1, gap, reverse))
	sg('send_command', 'attach_preproc TRAIN')

	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('send_command', 'convert TEST STRING CHAR STRING WORD %d %d %d %c' % (order, order-1, gap, reverse))
	sg('send_command', 'attach_preproc TEST')

	sg('send_command', 'set_kernel COMMSTRING WORD %d %d %s' % (size_cache, use_sign, normalization))
	sg('send_command', 'init_kernel TRAIN')
	km=sg('get_kernel_matrix')

	sg('send_command', 'init_kernel TEST')
	km=sg('get_kernel_matrix')

def sort_ulong_string ():
	print 'CommUlongString'

	data=get_dna()
	size_cache=10
	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true
	use_sign=False
	normalization='FULL'

	sg('send_command', 'add_preproc SORTULONGSTRING')
	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('send_command', 'convert TRAIN STRING CHAR STRING ULONG %d %d %d %c' % (order, order-1, gap, reverse))
	sg('send_command', 'attach_preproc TRAIN')

	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('send_command', 'convert TEST STRING CHAR STRING ULONG %d %d %d %c' % (order, order-1, gap, reverse))
	sg('send_command', 'attach_preproc TEST')

	sg('send_command', 'set_kernel COMMSTRING ULONG %d %d %s' % (size_cache, use_sign, normalization))
	sg('send_command', 'init_kernel TRAIN')
	km=sg('get_kernel_matrix')

	sg('send_command', 'init_kernel TEST')
	km=sg('get_kernel_matrix')

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
