#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'

parameter_list = [[traindat,3],[traindat,4]]

def clustering_kmeans_modular (fm_train=traindat,k=3):
	from modshogun import EuclideanDistance, RealFeatures, KMeansLloyd, Math_init_random, CSVFile
	Math_init_random(17)

	feats_train=RealFeatures(CSVFile(fm_train))
	distance=EuclideanDistance(feats_train, feats_train)

	kmeans=KMeansLloyd(k, distance)
	kmeans.train()

	out_centers = kmeans.get_cluster_centers()
	kmeans.get_radiuses()

	return out_centers, kmeans

if __name__=='__main__':
	print('KMeans')
	clustering_kmeans_modular(*parameter_list[0])

