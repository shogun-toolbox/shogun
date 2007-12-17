#!/usr/bin/env python
"""
Explicit examples on how to use clustering
"""

from numpy import array, concatenate
from numpy.random import rand, seed, permutation
from shogun.Distance import EuclidianDistance
from shogun.Features import RealFeatures
from shogun.Clustering import *

def get_cloud (num, num_vec, len):
	data=[rand(num_vec, len)+x/2 for x in xrange(num)]
	cloud=concatenate(data, axis=1)
	return array([permutation(x) for x in cloud])

def kmeans ():
	print 'KMeans'

	num_vec=11
	k=3
	data=get_cloud(k, num_vec, len=11)
	feats_train=RealFeatures(data)
	data=get_cloud(k, num_vec, len=17)
	feats_test=RealFeatures(data)
	distance=EuclidianDistance(feats_train, feats_train)

	kmeans=KMeans(k, distance)
	kmeans.train()

	distance.init(feats_train, feats_test)
	kmeans.get_centers()
	kmeans.get_radi()

def hierarchical ():
	print 'Hierarchical'

	num_vec=5
	merges=3
	data=get_cloud(merges, num_vec, 11)
	feats_train=RealFeatures(data)
	data=get_cloud(merges, num_vec, 17)
	feats_test=RealFeatures(data)
	distance=EuclidianDistance(feats_train, feats_train)

	hierarchical=Hierarchical(merges, distance)
	hierarchical.train()

	distance.init(feats_train, feats_test)
	hierarchical.get_merge_distance()
	hierarchical.get_pairs()

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	seed(42)

	kmeans()
	hierarchical()
