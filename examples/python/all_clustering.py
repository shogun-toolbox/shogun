#!/usr/bin/env python
"""
Explicit examples on how to use clustering
"""

from numpy import array, concatenate, sign, double
from numpy.random import rand, seed, permutation
from sg import sg

from tools.load import load_features, load_labels
fm_train_real=load_features('../data/fm_train_real.dat')


def kmeans ():
	print 'KMeans'

	size_cache=10
	k=3
	iter=1000

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_distance', 'EUCLIDIAN', 'REAL')
	sg('init_distance', 'TRAIN')
	sg('new_clustering', 'KMEANS')
	sg('train_clustering', k, iter)

	[radi, centers]=sg('get_clustering')


def hierarchical ():
	print 'Hierarchical'

	size_cache=10
	merges=3

	sg('set_features', 'TRAIN', fm_train_real)
	sg('set_distance', 'EUCLIDIAN', 'REAL')
	sg('init_distance', 'TRAIN')
	sg('new_clustering', 'HIERARCHICAL')
	sg('train_clustering', merges)

	[merge_distance, pairs]=sg('get_clustering')

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	kmeans()
	hierarchical()
