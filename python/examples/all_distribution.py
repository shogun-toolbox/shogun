#!/usr/bin/env python
"""
Explicit examples on how to use distributions
"""

from numpy import array, floor, ushort, ceil, concatenate, ones, zeros
from numpy.random import randint, seed, rand, permutation
from sg import sg

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
	reverse='n' # bit silly to not use boolean, set 'r' to yield true

#	sg('send_command', 'new_distribution HISTOGRAM')
	sg('send_command', 'add_preproc SORTWORDSTRING')

	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('send_command', 'convert TRAIN STRING CHAR STRING WORD %d %d %d %c' % (order, order-1, gap, reverse))
	sg('send_command', 'attach_preproc TRAIN')

#	sg('send_command', 'train_distribution')
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

	data=get_dna()
	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true

#	sg('send_command', 'new_distribution LinearHMM')
	sg('send_command', 'add_preproc SORTWORDSTRING')

	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('send_command', 'convert TRAIN STRING CHAR STRING WORD %d %d %d %c' % (order, order-1, gap, reverse))
	sg('send_command', 'attach_preproc TRAIN')

#	sg('send_command', 'train_distribution')
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
	hmms=list()
	liks=list()
	sequence=get_cubes()

	sg('send_command', 'new_hmm %d %d' % (N, M))
	sg('set_features', 'TRAIN', sequence, 'CUBE')
	sg('send_command', 'convert TRAIN STRING CHAR STRING WORD 1')
	sg('send_command', 'bw')
	hmm=sg('get_hmm')

	sg('send_command', 'new_hmm %d %d' % (N, M))
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

