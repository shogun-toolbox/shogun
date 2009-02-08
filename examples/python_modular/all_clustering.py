#!/usr/bin/env python
"""
Explicit examples on how to use clustering
"""

from numpy import array, concatenate
from numpy.random import rand, seed, permutation
from shogun.Distance import EuclidianDistance
from shogun.Features import RealFeatures
from shogun.Clustering import *

from tools.load import LoadMatrix
lm=LoadMatrix()
fm_train=lm.load_numbers('../data/fm_train_real.dat')


def kmeans ():
	print 'KMeans'

	k=3
	feats_train=RealFeatures(fm_train)
	distance=EuclidianDistance(feats_train, feats_train)

	kmeans=KMeans(k, distance)
	kmeans.train()

	kmeans.get_cluster_centers()
	kmeans.get_radiuses()


def hierarchical ():
	print 'Hierarchical'

	merges=3
	feats_train=RealFeatures(fm_train)
	distance=EuclidianDistance(feats_train, feats_train)

	hierarchical=Hierarchical(merges, distance)
	hierarchical.train()

	hierarchical.get_merge_distances()
	hierarchical.get_cluster_pairs()

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	kmeans()
	hierarchical()
