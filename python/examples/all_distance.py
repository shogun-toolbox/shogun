#!/usr/bin/env python
"""
Explicit examples on how to use the different distances
"""

from numpy import array, floor
from numpy.random import seed, rand
from sg import sg

def get_dna (len_seq_test_add=0):
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
		for j in range(60+len_seq_test_add):
			str1.append(acgt[floor(len_acgt*rand())])
	rand_test.append(''.join(str1))

	return {'train': rand_train, 'test': rand_test}

###########################################################################
# real features
###########################################################################

def euclidian_distance ():
	print 'EuclidianDistance'

	num_feats=10
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)

	sg('set_distance', 'EUCLIDIAN', 'REAL')

	sg('set_features', 'TRAIN', traindata)
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', testdata)
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

def canberra_metric ():
	print 'CanberraMetric'

	num_feats=10
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)

	sg('set_distance', 'CANBERRA', 'REAL')

	sg('set_features', 'TRAIN', traindata)
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', testdata)
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

def chebyshew_metric ():
	print 'ChebyshewMetric'

	num_feats=10
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)

	sg('set_distance', 'CHEBYSHEW', 'REAL')

	sg('set_features', 'TRAIN', traindata)
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', testdata)
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

def geodesic_metric ():
	print 'GeodesicMetric'

	num_feats=10
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)

	sg('set_distance', 'GEODESIC', 'REAL')

	sg('set_features', 'TRAIN', traindata)
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', testdata)
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

def jensen_metric ():
	print 'JensenMetric'

	num_feats=10
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)

	sg('set_distance', 'JENSEN', 'REAL')

	sg('set_features', 'TRAIN', traindata)
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', testdata)
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

def manhattan_metric ():
	print 'ManhattanMetric'

	num_feats=10
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)

	sg('set_distance', 'MANHATTAN', 'REAL')

	sg('set_features', 'TRAIN', traindata)
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', testdata)
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

def minkowski_metric ():
	print 'MinkowskiMetric'

	num_feats=10
	k=3.
	traindata=rand(num_feats, 11)
	testdata=rand(num_feats, 17)

	sg('set_distance', 'MINKOWSKI', 'REAL', k)

	sg('set_features', 'TRAIN', traindata)
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', testdata)
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

###########################################################################
# complex string features
###########################################################################

def canberra_word_distance ():
	print 'CanberraWordDistance'

	data=get_dna(len_seq_test_add=12)
	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true

	sg('set_distance', 'CANBERRA', 'WORD')
	sg('add_preproc', 'SORTWORDSTRING')

	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TEST')
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

def hamming_word_distance ():
	print 'HammingWordDistance'

	data=get_dna(len_seq_test_add=12)
	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true

	sg('set_distance', 'HAMMING', 'WORD')
	sg('add_preproc', 'SORTWORDSTRING')

	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TEST')
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

def manhattan_word_distance ():
	print 'ManhattanWordDistance'

	data=get_dna(len_seq_test_add=12)
	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true

	sg('set_distance', 'MANHATTAN', 'WORD')
	sg('add_preproc', 'SORTWORDSTRING')

	sg('set_features', 'TRAIN', data['train'], 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', data['test'], 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TEST')
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	seed(42)

	euclidian_distance()
	canberra_metric()
	chebyshew_metric()
	geodesic_metric()
	jensen_metric()
	manhattan_metric()
	minkowski_metric()

	canberra_word_distance()
	hamming_word_distance()
	manhattan_word_distance()
