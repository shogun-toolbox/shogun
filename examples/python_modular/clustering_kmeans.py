#!/usr/bin/env python
"""
Explicit examples on how to use clustering
"""

def kmeans ():
	print 'KMeans'

	from shogun.Distance import EuclidianDistance
	from shogun.Features import RealFeatures
	from shogun.Clustering import KMeans

	k=3
	feats_train=RealFeatures(fm_train)
	distance=EuclidianDistance(feats_train, feats_train)

	kmeans=KMeans(k, distance)
	kmeans.train()

	kmeans.get_cluster_centers()
	kmeans.get_radiuses()

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train=lm.load_numbers('../data/fm_train_real.dat')
	kmeans()

