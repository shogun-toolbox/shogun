#!/usr/bin/env python
##!/usr/bin/env python
#"""
#Explicit examples on how to use clustering
#"""
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[traindat,3],[traindat,4]]

def clustering_kmeans_modular (fm_train=traindat,k=3):

	from modshogun import EuclideanDistance
	from modshogun import RealFeatures
	from modshogun import KMeans
	from modshogun import Math_init_random
	Math_init_random(17)

	feats_train=RealFeatures(fm_train)
	distance=EuclideanDistance(feats_train, feats_train)

	kmeans=KMeans(k, distance)
	kmeans.train()

	out_centers = kmeans.get_cluster_centers()
	kmeans.get_radiuses()

	return out_centers, kmeans

if __name__=='__main__':
	print('KMeans')
	clustering_kmeans_modular(*parameter_list[0])

