#!/usr/bin/env python
"""
Explicit examples on how to use the different distances
"""

from numpy import char, array, floor
from numpy.random import seed, rand
from shogun.Features import *
from shogun.Distance import *
from shogun.PreProc import SortWordString

from tools.load import LoadMatrix
lm=LoadMatrix()
fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')


###########################################################################
# real features
###########################################################################

def bray_curtis_distance ():
	print 'BrayCurtisDistance'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	distance=BrayCurtisDistance(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def euclidian_distance ():
	print 'EuclidianDistance'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	distance=EuclidianDistance(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def norm_squared_distance ():
	print 'EuclidianDistance - NormSquared'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	distance=EuclidianDistance(feats_train, feats_train)
	distance.set_disable_sqrt(True)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def canberra_metric ():
	print 'CanberaMetric'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	distance=CanberraMetric(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def chebyshew_metric ():
	print 'ChebyshewMetric'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	distance=ChebyshewMetric(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def chi_square_distance ():
	print 'ChiSquareDistance'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	distance=ChiSquareDistance(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def cosine_distance ():
	print 'CosineDistance'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	distance=CosineDistance(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def geodesic_metric ():
	print 'GeodesicMetric'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	distance=GeodesicMetric(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def jensen_metric ():
	print 'JensenMetric'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	distance=JensenMetric(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def manhattan_metric ():
	print 'ManhattanMetric'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	distance=ManhattanMetric(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def minkowski_metric ():
	print 'MinkowskiMetric'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	k=3

	distance=MinkowskiMetric(feats_train, feats_train, k)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def tanimoto_distance ():
	print 'TanimotoDistance'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	distance=TanimotoDistance(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

def sparse_euclidian_distance ():
	print 'SparseEuclidianDistance'

	realfeat=RealFeatures(fm_train_real)
	feats_train=SparseRealFeatures()
	feats_train.obtain_from_simple(realfeat)
	realfeat=RealFeatures(fm_test_real)
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

	order=3
	gap=0
	reverse=False

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_train_dna)
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=SortWordString()
	preproc.init(feats_train)
	feats_train.add_preproc(preproc)
	feats_train.apply_preproc()

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_test_dna)
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

	order=3
	gap=0
	reverse=False

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_train_dna)
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=SortWordString()
	preproc.init(feats_train)
	feats_train.add_preproc(preproc)
	feats_train.apply_preproc()

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_test_dna)
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

	order=3
	gap=0
	reverse=False

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_train_dna)
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=SortWordString()
	preproc.init(feats_train)
	feats_train.add_preproc(preproc)
	feats_train.apply_preproc()

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_test_dna)
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

	bray_curtis_distance()
	euclidian_distance()
	norm_squared_distance()
	canberra_metric()
	chebyshew_metric()
	chi_square_distance()
	cosine_distance()
	geodesic_metric()
	jensen_metric()
	manhattan_metric()
	minkowski_metric()
	tanimoto_distance()

	sparse_euclidian_distance()

	canberra_word_distance()
	hamming_word_distance()
	manhattan_word_distance()
