#!/usr/bin/env python
"""
Explicit examples on how to use the different kernels
"""

from sys import maxint
from numpy import byte, ubyte, ushort, double, int, ones, zeros, sum, floor, array, arange, ceil, concatenate, sign
from numpy.random import randint, rand, seed, permutation
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
# byte features
###########################################################################

# LinearByte is b0rked
def linear_byte ():
	print 'LinearByte'

	num_feats=11
	traindata=randint(0, maxint, (num_feats, 11)).astype(byte)
	testdata=randint(0, maxint, (num_feats, 17)).astype(byte)

	sg('set_features', 'TRAIN', traindata, 'RAWBYTE')
	sg('set_features', 'TEST', testdata, 'RAWBYTE')
	sg('set_kernel', 'LINEAR BYTE', 10)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

###########################################################################
# real features
###########################################################################

def chi2 ():
	print 'Chi2'

	num_feats=11
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)
	width=1.4
	size_cache=10

	sg('set_features', 'TRAIN', traindata)
	sg('set_features', 'TEST', testdata)
	sg('set_kernel', 'CHI2', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def const ():
	print 'Const'

	num_feats=11
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)
	c=23.
	size_cache=10

	sg('set_features', 'TRAIN', traindata)
	sg('set_features', 'TEST', testdata)
	sg('set_kernel', 'CONST', 'REAL', size_cache, c)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def diag ():
	print 'Diag'

	num_feats=11
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)
	diag=23.
	size_cache=10

	sg('set_features', 'TRAIN', traindata)
	sg('set_features', 'TEST', testdata)
	sg('set_kernel', 'DIAG', 'REAL', size_cache, diag)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def gaussian ():
	print 'Gaussian'

	num_feats=11
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)
	width=1.9
	size_cache=10

	sg('set_features', 'TRAIN', traindata)
	sg('set_features', 'TEST', testdata)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def gaussian_shift ():
	print 'GaussianShift'

	num_feats=11
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)
	width=1.9
	max_shift=2
	shift_step=1
	size_cache=10

	sg('set_features', 'TRAIN', traindata)
	sg('set_features', 'TEST', testdata)
	sg('set_kernel', 'GAUSSIANSHIFT', 'REAL', size_cache, width, max_shift, shift_step)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def linear ():
	print 'Linear'

	num_feats=11
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)
	scale=1.2
	size_cache=10

	sg('set_features', 'TRAIN', traindata)
	sg('set_features', 'TEST', testdata)
	sg('set_kernel', 'LINEAR', 'REAL', size_cache, scale)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def poly ():
	print 'Poly'

	num_feats=11
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)
	degree=4
	inhomogene=False
	use_normalization=True
	size_cache=10

	sg('set_features', 'TRAIN', traindata)
	sg('set_features', 'TEST', testdata)
	sg('set_kernel', 'POLY', 'REAL', size_cache, degree, inhomogene, use_normalization)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def sigmoid ():
	print 'Sigmoid'

	num_feats=11
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)
	gamma=1.2
	coef0=1.3
	size_cache=10

	sg('set_features', 'TRAIN', traindata)
	sg('set_features', 'TEST', testdata)
	sg('set_kernel', 'SIGMOID', 'REAL', size_cache, gamma, coef0)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

###########################################################################
# word features
###########################################################################

def linear_word ():
	print 'LinearWord'

	maxval=2**16-1
	num_feats=11
	traindata=randint(0, maxval, (num_feats, 11)).astype(ushort)
	testdata=randint(0, maxval, (num_feats, 17)).astype(ushort)
	size_cache=10
	scale=1.4

	sg('set_features', 'TRAIN', traindata)
	sg('set_features', 'TEST', testdata)
	sg('set_kernel', 'LINEAR', 'WORD', size_cache, scale)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def poly_match_word ():
	print 'PolyMatchWord'

	maxval=2**16-1
	num_feats=11
	traindata=randint(0, maxval, (num_feats, 11)).astype(ushort)
	testdata=randint(0, maxval, (num_feats, 17)).astype(ushort)
	size_cache=10
	degree=2
	inhomogene=True
	normalize=True

	sg('set_features', 'TRAIN', traindata)
	sg('set_features', 'TEST', testdata)
	sg('set_kernel', 'POLYMATCH', 'WORD', size_cache, degree, inhomogene, normalize)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

###########################################################################
# string features
###########################################################################

def fixed_degree_string ():
	print 'FixedDegreeString'

	data=get_dna()
	size_cache=10
	degree=3

	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('set_kernel', 'FIXEDDEGREE', 'CHAR', size_cache, degree)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def linear_string ():
	print 'LinearString'

	data=get_dna()
	size_cache=10

	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('set_kernel', 'LINEAR', 'CHAR', size_cache)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def local_alignment_string():
	print 'LocalAlignmentString'

	data=get_dna()
	size_cache=10

	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('set_kernel', 'LOCALALIGNMENT', 'CHAR', size_cache)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def poly_match_string ():
	print 'PolyMatchString'

	data=get_dna()
	size_cache=10
	degree=3
	inhomogene=False

	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('set_kernel', 'POLYMATCH', 'CHAR', size_cache, degree, inhomogene)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def weighted_degree_string ():
	print 'WeightedDegreeString'

	data=get_dna()
	size_cache=10
	degree=20

	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', size_cache, degree)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def weighted_degree_position_string ():
	print 'WeightedDegreePositionString'

	data=get_dna()
	size_cache=10
	degree=20

	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('set_kernel', 'WEIGHTEDDEGREEPOS', 'CHAR', size_cache, degree)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def locality_improved_string ():
	print 'LocalityImprovedString'

	data=get_dna()
	size_cache=10
	length=5
	inner_degree=5
	outer_degree=inner_degree+2

	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('set_kernel', 'LIK', 'CHAR', size_cache, length, inner_degree, outer_degree)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def simple_locality_improved_string ():
	print 'SimpleLocalityImprovedString'

	data=get_dna()
	size_cache=10
	length=5
	inner_degree=5
	outer_degree=inner_degree+2

	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('set_kernel', 'SLIK', 'CHAR', size_cache, length, inner_degree, outer_degree)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

###########################################################################
# complex string features
###########################################################################

def comm_word_string ():
	print 'CommWordString'

	data=get_dna()
	size_cache=10
	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true
	use_sign=False
	normalization='FULL'

	sg('add_preproc', 'SORTWORDSTRING')
	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')

	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TEST')

	sg('set_kernel', 'COMMSTRING', 'WORD', size_cache, use_sign, normalization)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def weighted_comm_word_string ():
	print 'WeightedCommWordString'

	data=get_dna()
	size_cache=10
	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true
	use_sign=False
	normalization='FULL'

	sg('add_preproc', 'SORTWORDSTRING')
	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')

	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TEST')

	sg('set_kernel', 'WEIGHTEDCOMMSTRING', 'WORD', size_cache, use_sign, normalization)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def comm_ulong_string ():
	print 'CommUlongString'

	data=get_dna()
	size_cache=10
	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true
	use_sign=False
	normalization='FULL'

	sg('add_preproc', 'SORTULONGSTRING')
	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')

	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'ULONG', order, order-1, gap, reverse)
	sg('attach_preproc', 'TEST')

	sg('set_kernel', 'COMMSTRING', 'ULONG', size_cache, use_sign, normalization)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

###########################################################################
# misc kernels
###########################################################################

def distance ():
	print 'Distance'

	num_feats=10
	traindata=rand(num_feats, 9)
	testdata=rand(num_feats, 19)
	width=1.7
	size_cache=10

	sg('set_features', 'TRAIN', traindata)
	sg('set_features', 'TEST', testdata)
	sg('set_distance', 'EUCLIDIAN', 'REAL')
	sg('set_kernel', 'DISTANCE', size_cache, width)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')


def combined ():
	print 'Combined'

	num_feats=10
	traindata=rand(num_feats, 9)
	testdata=rand(num_feats, 19)
	size_cache=10
	weight=1.

	sg('set_kernel', 'COMBINED', size_cache)
	sg('add_kernel', weight, 'LINEAR', 'REAL', size_cache)
	sg('add_features', 'TRAIN', traindata)
	sg('add_features', 'TEST', testdata)
	sg('add_kernel', weight, 'GAUSSIAN', 'REAL', size_cache, 1.)
	sg('add_features', 'TRAIN', traindata)
	sg('add_features', 'TEST', testdata)
	sg('add_kernel', weight, 'POLY', 'REAL', size_cache, 3, False)
	sg('add_features', 'TRAIN', traindata)
	sg('add_features', 'TEST', testdata)

	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def plugin_estimate ():
	print 'PluginEstimate w/ HistogramWord'

	data=get_dna()
	size_cache=10
	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true
	use_sign=False
	normalization='FULL'

	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)

	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)

	labels=sign(rand(1,11)-0.5)[0]
	pseudo_pos=1e-1
	pseudo_neg=1e-1
	sg('new_plugin_estimator', pseudo_pos, pseudo_neg)
	sg('set_labels', 'TRAIN', labels)
	sg('train_estimator')

	sg('set_kernel', 'HISTOGRAM', 'WORD', size_cache)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
# not supported yet
#	lab=sg('plugin_estimate_classify')
	km=sg('get_kernel_matrix')

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	seed(42)

#	linear_byte()

	chi2()
	const()
	diag()
	gaussian()
	gaussian_shift()
	linear()
	poly()
	sigmoid()

	linear_word()
	poly_match_word()

	fixed_degree_string()
	linear_string()
	local_alignment_string()
	poly_match_string()
	weighted_degree_string()
	weighted_degree_position_string()
	locality_improved_string()
	simple_locality_improved_string()

	comm_word_string()
	weighted_comm_word_string()
	comm_ulong_string()

	distance()
	combined()
	plugin_estimate()
