#!/usr/bin/env python
"""
Explicit examples on how to use distributions
"""

from numpy import array, floor, ushort
from numpy.random import randint, seed, rand
from shogun.Features import StringWordFeatures, CharFeatures, StringCharFeatures
from shogun.PreProc import SortWordString
from shogun.Distribution import *

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
# distributions
###########################################################################

def histogram ():
	print 'Histogram'

	data=get_dna()
	order=3
	gap=0
	reverse=False

	charfeat=StringCharFeatures(DNA)
	charfeat.set_string_features(data['train'])
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

	histo.get_log_likelihood_sample()

def linear_hmm ():
	print 'LinearHMM'

	data=get_dna()
	order=3
	gap=0
	reverse=False

	charfeat=StringCharFeatures(DNA)
	charfeat.set_string_features(data['train'])
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

	hmm.get_log_likelihood_sample()

def hmm ():
	print 'HMM'

	data=get_dna()
	feats_train=StringCharFeatures(DNA)
	feats_train.set_string_features(data['train'])
	feats_test=StringCharFeatures(DNA)
	feats_test.set_string_features(data['test'])
	N=1
	M=2
	model=Model()
	pseudo=1.

	hmm=HMM(N, M, model, pseudo)
	#hmm.set_observations(feats_train)
	hmm.train()

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	seed(42)

	histogram()
	linear_hmm()
	hmm()

