#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')

parameter_list = [[traindat,3],[traindat,4]]

def clustering_hierarchical_modular (fm_train=traindat,merges=3):

	from modshogun import EuclideanDistance
	from modshogun import RealFeatures
	from modshogun import Hierarchical


	feats_train=RealFeatures(fm_train)
	distance=EuclideanDistance(feats_train, feats_train)

	hierarchical=Hierarchical(merges, distance)
	hierarchical.train()

	out_distance = hierarchical.get_merge_distances()
	out_cluster = hierarchical.get_cluster_pairs()

	return hierarchical,out_distance,out_cluster 

if __name__=='__main__':
	print('Hierarchical')
	clustering_hierarchical_modular(*parameter_list[0])
