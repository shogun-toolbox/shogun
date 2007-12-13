#!/usr/bin/env python
"""
Explicit examples on how to use distributions
"""

from numpy import array, floor, ushort
from numpy.random import randint, seed, rand
from shogun.Features import WordFeatures, CharFeatures, StringCharFeatures
from shogun.Distribution import *

def get_dna ():
	acgt=array(['A', 'C', 'G','T'])
	len_acgt=len(acgt)
	dtrain=[]
	dtest=[]

	for i in xrange(11):
		str1=[]
		str2=[]
		for j in range(60):
			str1.append(acgt[floor(len_acgt*rand())])
			str2.append(acgt[floor(len_acgt*rand())])
		dtrain.append(''.join(str1))
	dtest.append(''.join(str2))
	
	for i in xrange(6):
		str1=[]
		for j in range(60):
			str1.append(acgt[floor(len_acgt*rand())])
	dtest.append(''.join(str1))

	return {'train': dtrain, 'test': dtest}

###########################################################################
# distributions
###########################################################################

def histogram ():
	print 'Histogram'

	maxval=2**16-1
	rows=11
	data=randint(0, maxval, (rows, 11)).astype(ushort)
	feats=WordFeatures(data)

	histo=Histogram(feats)
	histo.train()

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
	hmm()

