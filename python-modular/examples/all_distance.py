#!/usr/bin/env python
"""
Explicit examples on how to use the different distances
"""

from numpy import array, floor
from numpy.random import seed, rand
from shogun.Features import *
from shogun.Distance import *
from shogun.PreProc import SortWordString

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
# real features
###########################################################################

def euclidian_distance ():
	print 'EuclidianDistance'

	rows=9
	data=rand(rows, 11)
	feats_train=RealFeatures(data)
	data=rand(rows, 19)
	feats_test=RealFeatures(data)

	distance=EuclidianDistance(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def canberra_metric ():
	print 'CanberaMetric'

	rows=9
	data=rand(rows, 10)
	feats_train=RealFeatures(data)
	data=rand(rows, 17)
	feats_test=RealFeatures(data)

	distance=CanberraMetric(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def chebyshew_metric ():
	print 'ChebyshewMetric'

	rows=9
	data=rand(rows, 11)
	feats_train=RealFeatures(data)
	data=rand(rows, 17)
	feats_test=RealFeatures(data)

	distance=ChebyshewMetric(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def geodesic_metric ():
	print 'GeodesicMetric'

	rows=9
	data=rand(rows, 11)
	feats_train=RealFeatures(data)
	data=rand(rows, 21)
	feats_test=RealFeatures(data)

	distance=GeodesicMetric(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def jensen_metric ():
	print 'JensenMetric'

	rows=9
	data=rand(rows, 11)
	feats_train=RealFeatures(data)
	data=rand(rows, 17)
	feats_test=RealFeatures(data)

	distance=JensenMetric(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def manhattan_metric ():
	print 'ManhattanMetric'

	rows=9
	data=rand(rows, 11)
	feats_train=RealFeatures(data)
	data=rand(rows, 17)
	feats_test=RealFeatures(data)

	distance=ManhattanMetric(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def minkowski_metric ():
	print 'MinkowskiMetric'

	rows=9
	data=rand(rows, 11)
	feats_train=RealFeatures(data)
	data=rand(rows, 15)
	feats_test=RealFeatures(data)
	k=3

	distance=MinkowskiMetric(feats_train, feats_train, k)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def sparse_euclidian_distance ():
	print 'SparseEuclidianDistance'

	rows=11
	data=rand(rows, 11)
	realfeat=RealFeatures(data)
	feats_train=SparseRealFeatures()
	feats_train.obtain_from_simple(realfeat)
	data=rand(rows, 17)
	realfeat=RealFeatures(data)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)

	distance=SparseEuclidianDistance(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()


###########################################################################
# complex string features
###########################################################################

def canberra_word_distance ():
	print 'CanberraWordDistance'

	data=get_dna()
	order=3
	gap=0
	reverse=False

	charfeat=StringCharFeatures(DNA)
	charfeat.set_string_features(data['train'])
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=SortWordString()
	preproc.init(feats_train)
	feats_train.add_preproc(preproc)
	feats_train.apply_preproc()

	charfeat=StringCharFeatures(DNA)
	charfeat.set_string_features(data['test'])
	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats_test.add_preproc(preproc)
	feats_test.apply_preproc()

	distance=CanberraWordDistance(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def hamming_word_distance ():
	print 'HammingWordDistance'

	data=get_dna()
	order=3
	gap=0
	reverse=False

	charfeat=StringCharFeatures(DNA)
	charfeat.set_string_features(data['train'])
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=SortWordString()
	preproc.init(feats_train)
	feats_train.add_preproc(preproc)
	feats_train.apply_preproc()

	charfeat=StringCharFeatures(DNA)
	charfeat.set_string_features(data['test'])
	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats_test.add_preproc(preproc)
	feats_test.apply_preproc()

	use_sign=False

	distance=HammingWordDistance(feats_train, feats_train, use_sign)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def manhattan_word_distance ():
	print 'ManhattanWordDistance'

	data=get_dna()
	order=3
	gap=0
	reverse=False

	charfeat=StringCharFeatures(DNA)
	charfeat.set_string_features(data['train'])
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=SortWordString()
	preproc.init(feats_train)
	feats_train.add_preproc(preproc)
	feats_train.apply_preproc()

	charfeat=StringCharFeatures(DNA)
	charfeat.set_string_features(data['test'])
	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats_test.add_preproc(preproc)
	feats_test.apply_preproc()

	distance=ManhattanWordDistance(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

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

	sparse_euclidian_distance()

	canberra_word_distance()
	hamming_word_distance()
	manhattan_word_distance()
