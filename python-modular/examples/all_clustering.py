#!/usr/bin/env python
"""
Explicit examples on how to use clustering
"""

from numpy.random import rand, seed
from shogun.Distance import EuclidianDistance
from shogun.Features import RealFeatures
from shogun.Clustering import *

def kmeans ():
	print 'KMeans'

	rows=11
	data=rand(rows, 11)
	feats_train=RealFeatures(data)
	data=rand(rows, 17)
	feats_test=RealFeatures(data)
	k=3
	distance=EuclidianDistance(feats_train, feats_train)

	kmeans=KMeans(k, distance)
	kmeans.train()

	distance.init(feats_train, feats_test)
	#kmeans.classify().get_labels()

def hierarchical ():
	print 'Hierarchical'

	rows=5
	data=rand(rows, 11)
	feats_train=RealFeatures(data)
	data=rand(rows, 17)
	feats_test=RealFeatures(data)
	merges=3
	distance=EuclidianDistance(feats_train, feats_train)

	hierarchical=Hierarchical(merges, distance)
	hierarchical.train()

	distance.init(feats_train, feats_test)
	#hierarchical.classify().get_labels()

###########################################################################
# call functions
###########################################################################

if __name__=='__main__':
	seed(42)

	kmeans()
	hierarchical()
