#!/usr/bin/env python
"""
Explicit examples on how to use distributions
"""

from numpy import array, floor, char, ushort, ceil, concatenate, ones, zeros
from numpy.random import randint, seed, rand, permutation
from shogun.Features import StringWordFeatures, CharFeatures, StringCharFeatures, DNA, CUBE
from shogun.PreProc import SortWordString
from shogun.Distribution import *

from tools.load import LoadMatrix
lm=LoadMatrix()
fm_dna=lm.load_dna('../data/fm_train_dna.dat')
fm_cube=lm.load_cubes('../data/fm_train_cube.dat')

###########################################################################
# distributions
###########################################################################

def histogram ():
	print 'Histogram'

	order=3
	gap=0
	reverse=False

	charfeat=StringCharFeatures(DNA)
	charfeat.set_string_features(fm_dna)
	feats=StringWordFeatures(charfeat.get_alphabet())
	feats.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=SortWordString()
	preproc.init(feats)
	feats.add_preproc(preproc)
	feats.apply_preproc()

	histo=Histogram(feats)
	histo.train()

	histo.get_histogram()

	num_examples=feats.get_num_vectors()
	num_param=histo.get_num_model_parameters()
	for i in xrange(num_examples):
		for j in xrange(num_param):
			histo.get_log_derivative(j, i)

	histo.get_log_likelihood()
	histo.get_log_likelihood_sample()

def linear_hmm ():
	print 'LinearHMM'

	order=3
	gap=0
	reverse=False

	charfeat=StringCharFeatures(DNA)
	charfeat.set_string_features(fm_dna)
	feats=StringWordFeatures(charfeat.get_alphabet())
	feats.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=SortWordString()
	preproc.init(feats)
	feats.add_preproc(preproc)
	feats.apply_preproc()

	hmm=LinearHMM(feats)
	hmm.train()

	hmm.get_transition_probs()

	num_examples=feats.get_num_vectors()
	num_param=hmm.get_num_model_parameters()
	for i in xrange(num_examples):
		for j in xrange(num_param):
			hmm.get_log_derivative(j, i)

	hmm.get_log_likelihood()
	hmm.get_log_likelihood_sample()

def hmm ():
	print 'HMM'

	N=3
	M=6
	pseudo=1e-1
	order=1
	gap=0
	reverse=False
	num_examples=2

	charfeat=StringCharFeatures(CUBE)
	charfeat.set_string_features(fm_cube)
	feats=StringWordFeatures(charfeat.get_alphabet())
	feats.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=SortWordString()
	preproc.init(feats)
	feats.add_preproc(preproc)
	feats.apply_preproc()

	hmm=HMM(feats, N, M, pseudo)
	hmm.train()
	hmm.baum_welch_viterbi_train(BW_NORMAL)

	num_examples=feats.get_num_vectors()
	num_param=hmm.get_num_model_parameters()
	for i in xrange(num_examples):
		for j in xrange(num_param):
			hmm.get_log_derivative(j, i)

	best_path=0
	best_path_state=0
	for i in xrange(num_examples):
		best_path+=hmm.best_path(i)
		for j in xrange(N):
			best_path_state+=hmm.get_best_path_state(i, j)

	hmm.get_log_likelihood()
	hmm.get_log_likelihood_sample()

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	seed(42)

	histogram()
	linear_hmm()
	hmm()

