#!/usr/bin/env python
"""
Explicit examples on how to use distributions
"""

from numpy import array, floor, ushort, ceil, concatenate, ones, zeros, double, char
from numpy.random import randint, seed, rand, permutation
from sg import sg

from tools.load import load_features, load_labels
fm_train_dna=load_features('../data/fm_train_dna.dat', char)
fm_test_dna=load_features('../data/fm_test_dna.dat', char)


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

	return sequence


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

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
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

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
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
	sequence=get_cubes()

	sg('new_hmm',N, M)
	sg('set_features', 'TRAIN', sequence, 'CUBE')
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

