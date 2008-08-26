#!/usr/bin/env python
"""
Explicit examples on how to use the different kernels
"""

from sys import maxint
from numpy import ubyte, ushort, double, int, char, ones, zeros, sum, floor, array, arange, ceil, concatenate, sign
from numpy.random import randint, rand, seed, permutation
from sg import sg

from tools.load import LoadMatrix
lm=LoadMatrix()
fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
fm_train_word=ushort(lm.load_numbers('../data/fm_test_word.dat'))
fm_test_word=ushort(lm.load_numbers('../data/fm_test_word.dat'))
fm_train_byte=ubyte(lm.load_numbers('../data/fm_train_byte.dat'))
fm_test_byte=ubyte(lm.load_numbers('../data/fm_test_byte.dat'))
fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')
label_train_dna=lm.load_labels('../data/label_train_dna.dat')


###########################################################################
# byte features
###########################################################################

# LinearByte is b0rked
def linear_byte ():
	print 'LinearByte'

	sg('set_features', 'TRAIN', fm_train_byte, 'RAWBYTE')
	sg('set_features', 'TEST', fm_test_byte, 'RAWBYTE')
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

	width=1.4
	size_cache=10

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_features', 'TEST', fm_test_real)
	sg('set_kernel', 'CHI2', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def const ():
	print 'Const'

	c=23.
	size_cache=10

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_features', 'TEST', fm_test_real)
	sg('set_kernel', 'CONST', 'REAL', size_cache, c)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def diag ():
	print 'Diag'

	diag=23.
	size_cache=10

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_features', 'TEST', fm_test_real)
	sg('set_kernel', 'DIAG', 'REAL', size_cache, diag)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def gaussian ():
	print 'Gaussian'

	width=1.9
	size_cache=10

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_features', 'TEST', fm_test_real)
	sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def gaussian_shift ():
	print 'GaussianShift'

	width=1.9
	max_shift=2
	shift_step=1
	size_cache=10

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_features', 'TEST', fm_test_real)
	sg('set_kernel', 'GAUSSIANSHIFT', 'REAL', size_cache, width, max_shift, shift_step)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def linear ():
	print 'Linear'

	scale=1.2
	size_cache=10

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_features', 'TEST', fm_test_real)
	sg('set_kernel', 'LINEAR', 'REAL', size_cache, scale)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def poly ():
	print 'Poly'

	degree=4
	inhomogene=False
	use_normalization=True
	size_cache=10

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_features', 'TEST', fm_test_real)
	sg('set_kernel', 'POLY', 'REAL', size_cache, degree, inhomogene, use_normalization)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def sigmoid ():
	print 'Sigmoid'

	num_feats=11
	fm_train_real=rand(num_feats, 11)
	fm_test_real=rand(num_feats, 17)
	gamma=1.2
	coef0=1.3
	size_cache=10

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_features', 'TEST', fm_test_real)
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

	size_cache=10
	scale=1.4

	sg('set_features', 'TRAIN', fm_train_word)
	sg('set_features', 'TEST', fm_test_word)
	sg('set_kernel', 'LINEAR', 'WORD', size_cache, scale)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def poly_match_word ():
	print 'PolyMatchWord'

	size_cache=10
	degree=2
	inhomogene=True
	normalize=True

	sg('set_features', 'TRAIN', fm_train_word)
	sg('set_features', 'TEST', fm_test_word)
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

	size_cache=10
	degree=3

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('set_kernel', 'FIXEDDEGREE', 'CHAR', size_cache, degree)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def linear_string ():
	print 'LinearString'

	size_cache=10

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('set_kernel', 'LINEAR', 'CHAR', size_cache)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def local_alignment_string():
	print 'LocalAlignmentString'

	size_cache=10

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('set_kernel', 'LOCALALIGNMENT', 'CHAR', size_cache)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def oligo_string ():
	print 'OligoString'

	size_cache=10
	k=3
	width=1.2

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('set_kernel', 'OLIGO', 'CHAR', size_cache, k, width)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def poly_match_string ():
	print 'PolyMatchString'

	size_cache=10
	degree=3
	inhomogene=False

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('set_kernel', 'POLYMATCH', 'CHAR', size_cache, degree, inhomogene)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def weighted_degree_string ():
	print 'WeightedDegreeString'

	size_cache=10
	degree=20

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', size_cache, degree)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def weighted_degree_position_string ():
	print 'WeightedDegreePositionString'

	size_cache=10
	degree=20

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('set_kernel', 'WEIGHTEDDEGREEPOS', 'CHAR', size_cache, degree)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def locality_improved_string ():
	print 'LocalityImprovedString'

	size_cache=10
	length=5
	inner_degree=5
	outer_degree=inner_degree+2

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('set_kernel', 'LIK', 'CHAR', size_cache, length, inner_degree, outer_degree)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def simple_locality_improved_string ():
	print 'SimpleLocalityImprovedString'

	size_cache=10
	length=5
	inner_degree=5
	outer_degree=inner_degree+2

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
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

def weighted_comm_word_string ():
	print 'WeightedCommWordString'

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

	sg('set_kernel', 'WEIGHTEDCOMMSTRING', 'WORD', size_cache, use_sign, normalization)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def comm_ulong_string ():
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
# misc kernels
###########################################################################

def distance ():
	print 'Distance'

	width=1.7
	size_cache=10

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_features', 'TEST', fm_test_real)
	sg('set_distance', 'EUCLIDIAN', 'REAL')
	sg('set_kernel', 'DISTANCE', size_cache, width)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')


def combined ():
	print 'Combined'

	size_cache=10
	weight=1.

	sg('set_kernel', 'COMBINED', size_cache)
	sg('add_kernel', weight, 'LINEAR', 'REAL', size_cache)
	sg('add_features', 'TRAIN', fm_train_real)
	sg('add_features', 'TEST', fm_test_real)
	sg('add_kernel', weight, 'GAUSSIAN', 'REAL', size_cache, 1.)
	sg('add_features', 'TRAIN', fm_train_real)
	sg('add_features', 'TEST', fm_test_real)
	sg('add_kernel', weight, 'POLY', 'REAL', size_cache, 3, False)
	sg('add_features', 'TRAIN', fm_train_real)
	sg('add_features', 'TEST', fm_test_real)

	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
	km=sg('get_kernel_matrix')

def plugin_estimate_histogram ():
	print 'PluginEstimate w/ HistogramWord'

	size_cache=10
	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true
	use_sign=False
	normalization='FULL'

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)

	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)

	pseudo_pos=1e-1
	pseudo_neg=1e-1
	sg('new_plugin_estimator', pseudo_pos, pseudo_neg)
	sg('set_labels', 'TRAIN', label_train_dna)
	sg('train_estimator')

	sg('set_kernel', 'HISTOGRAM', 'WORD', size_cache)
	sg('init_kernel', 'TRAIN')
	km=sg('get_kernel_matrix')

	sg('init_kernel', 'TEST')
# not supported yet
#	lab=sg('plugin_estimate_classify')
	km=sg('get_kernel_matrix')


def plugin_estimate_salzberg ():
	print 'PluginEstimate w/ SalzbergWord'

	size_cache=10
	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true
	use_sign=False
	normalization='FULL'

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)

	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)

	pseudo_pos=1e-1
	pseudo_neg=1e-1
	sg('new_plugin_estimator', pseudo_pos, pseudo_neg)
	sg('set_labels', 'TRAIN', label_train_dna)
	sg('train_estimator')

	sg('set_kernel', 'SALZBERG', 'WORD', size_cache)
	#sg('set_prior_probs', 0.4, 0.6)
	sg('set_prior_probs_from_labels', label_train_dna)
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
	oligo_string()
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
	plugin_estimate_histogram()
	plugin_estimate_salzberg()
