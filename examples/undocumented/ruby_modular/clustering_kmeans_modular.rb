# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
##!/usr/bin/env python
#"""
#Explicit examples on how to use clustering
#"""

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')

parameter_list = [[traindat,3],[traindat,4]]

def clustering_kmeans_modular(fm_train=traindat,k=3)

	Math_init_random(17)

# *** 	feats_train=RealFeatures(fm_train)
	feats_train=Modshogun::RealFeatures.new
	feats_train.set_features(fm_train)
# *** 	distance=EuclidianDistance(feats_train, feats_train)
	distance=Modshogun::EuclidianDistance.new
	distance.set_features(feats_train, feats_train)

# *** 	kmeans=KMeans(k, distance)
	kmeans=Modshogun::KMeans.new
	kmeans.set_features(k, distance)
	kmeans.train()

	out_centers = kmeans.get_cluster_centers()
	kmeans.get_radiuses()

	return out_centers, kmeans


end
if __FILE__ == $0
	puts 'KMeans'
	clustering_kmeans_modular(*parameter_list[0])


end
