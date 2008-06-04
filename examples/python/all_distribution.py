#!/usr/bin/env python
"""
Explicit examples on how to use distributions
"""

from numpy import array, floor, ushort, ceil, concatenate, ones, zeros, double, char
from numpy.random import randint, seed, rand, permutation
from sg import sg

from tools.load import LoadMatrix
lm=LoadMatrix()
fm_train=lm.load_dna('../data/fm_train_dna.dat')
fm_cube=lm.load_cubes('../data/fm_cube_train.dat')

###########################################################################
# distributions
###########################################################################

def histogram ():
	print 'Histogram'

	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true

#	sg('new_distribution', 'HISTOGRAM')
	sg('add_preproc', 'SORTWORDSTRING')

	sg('set_features', 'TRAIN', fm_train, 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')

#	sg('train_distribution')
#	histo=sg('get_histogram')

#	num_examples=11
#	num_param=sg('get_histogram_num_model_parameters')
#	for i in xrange(num_examples):
#		for j in xrange(num_param):
#			sg('get_log_derivative %d %d' % (j, i))

#	sg('get_log_likelihood')
#	sg('get_log_likelihood_sample')

def linear_hmm ():
	print 'LinearHMM'

	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true

#	sg('new_distribution', 'LinearHMM')
	sg('add_preproc', 'SORTWORDSTRING')

	sg('set_features', 'TRAIN', fm_train, 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')

#	sg('train_distribution')
#	histo=sg('get_histogram')

#	num_examples=11
#	num_param=sg('get_histogram_num_model_parameters')
#	for i in xrange(num_examples):
#		for j in xrange(num_param):
#			sg('get_log_derivative %d %d' % (j, i))

#	sg('get_log_likelihood')
#	sg('get_log_likelihood_sample')

def hmm ():
	print 'HMM'

	N=3
	M=6
	order=1
	hmms=list()
	liks=list()

	sg('new_hmm',N, M)
	sg('set_features', 'TRAIN', fm_cube, 'CUBE')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order)
	sg('bw')
	hmm=sg('get_hmm')

	sg('new_hmm', N, M)
	sg('set_hmm', hmm[0], hmm[1], hmm[2], hmm[3])
	likelihood=sg('hmm_likelihood')

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	seed(42)

	histogram()
	linear_hmm()
	hmm()

