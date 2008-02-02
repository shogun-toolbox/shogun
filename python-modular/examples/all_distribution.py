#!/usr/bin/env python
"""
Explicit examples on how to use distributions
"""

from numpy import array, floor, ushort, ceil, concatenate, ones, zeros
from numpy.random import randint, seed, rand, permutation
from shogun.Features import StringWordFeatures, CharFeatures, StringCharFeatures
from shogun.PreProc import SortWordString
from shogun.Distribution import *

def get_cubes (num=2):
	leng=50
	rep=5
	weight=1

	sequence=[]

	for i in xrange(num):
		# generate a sequence with characters 1-6 drawn from 3 loaded cubes
		loaded=[]
		for j in xrange(3):
			draw=[x*ones((1, ceil(leng*rand())), int)[0] \
				for x in xrange(1, 7)]
			loaded.append(permutation(concatenate(draw)))

		draws=[]
		for j in xrange(len(loaded)):
			data=ones((1, ceil(rep*rand())), int)
			draws=concatenate((j*data[0], draws))
		draws=permutation(draws)

		seq=[]
		for j in xrange(len(draws)):
			len_loaded=len(loaded[draws[j]])
			weighted=int(ceil(
				((1-weight)*rand()+weight)*len_loaded))
			perm=permutation(len_loaded)
			shuffled=[str(loaded[draws[j]][x]) for x in perm[:weighted]]
			seq=concatenate((seq, shuffled))

		sequence.append(''.join(seq))

	return {'train':sequence, 'test':sequence}

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

	histo.get_log_likelihood()
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
	data=get_cubes(num_examples)

	charfeat=StringCharFeatures(CUBE)
	charfeat.set_string_features(data['train'])
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

