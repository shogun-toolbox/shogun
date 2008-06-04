#!/usr/bin/env python
"""
Explicit examples on how to use the different distances
"""

from numpy import array, floor, double, char
from numpy.random import seed, rand
from sg import sg

from tools.load import LoadMatrix
lm=LoadMatrix()
fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
fm_train_dna=lm.load_dna('../data/fm_train_dna.dat')
fm_test_dna=lm.load_dna('../data/fm_test_dna.dat')


###########################################################################
# real features
###########################################################################

def euclidian_distance ():
	print 'EuclidianDistance'
	sg('set_distance', 'EUCLIDIAN', 'REAL')

	sg('set_features', 'TRAIN', fm_train_real)
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', fm_test_real)
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

def canberra_metric ():
	print 'CanberraMetric'
	sg('set_distance', 'CANBERRA', 'REAL')

	sg('set_features', 'TRAIN', fm_train_real)
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', fm_test_real)
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

def chebyshew_metric ():
	print 'ChebyshewMetric'
	sg('set_distance', 'CHEBYSHEW', 'REAL')

	sg('set_features', 'TRAIN', fm_train_real)
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', fm_test_real)
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

def geodesic_metric ():
	print 'GeodesicMetric'
	sg('set_distance', 'GEODESIC', 'REAL')

	sg('set_features', 'TRAIN', fm_train_real)
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', fm_test_real)
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

def jensen_metric ():
	print 'JensenMetric'
	sg('set_distance', 'JENSEN', 'REAL')

	sg('set_features', 'TRAIN', fm_train_real)
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', fm_test_real)
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

def manhattan_metric ():
	print 'ManhattanMetric'
	sg('set_distance', 'MANHATTAN', 'REAL')

	sg('set_features', 'TRAIN', fm_train_real)
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', fm_test_real)
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

def minkowski_metric ():
	print 'MinkowskiMetric'

	k=3.
	sg('set_distance', 'MINKOWSKI', 'REAL', k)

	sg('set_features', 'TRAIN', fm_train_real)
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', fm_test_real)
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

###########################################################################
# complex string features
###########################################################################

def canberra_word_distance ():
	print 'CanberraWordDistance'

	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true

	sg('set_distance', 'CANBERRA', 'WORD')
	sg('add_preproc', 'SORTWORDSTRING')

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TEST')
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

def hamming_word_distance ():
	print 'HammingWordDistance'

	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true

	sg('set_distance', 'HAMMING', 'WORD')
	sg('add_preproc', 'SORTWORDSTRING')

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TEST')
	sg('init_distance', 'TEST')
	dm=sg('get_distance_matrix')

def manhattan_word_distance ():
	print 'ManhattanWordDistance'

	order=3
	gap=0
	reverse='n' # bit silly to not use boolean, set 'r' to yield true

	sg('set_distance', 'MANHATTAN', 'WORD')
	sg('add_preproc', 'SORTWORDSTRING')

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')
	sg('init_distance', 'TRAIN')
	dm=sg('get_distance_matrix')

	sg('set_features', 'TEST', fm_test_dna, 'DNA')
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
