#!/usr/bin/env python
parameter_list = [[1000,2,8],[1000,4,8]]

import shogun as sg
from numpy import *
#from pylab import *

def run_clustering(data, k):

	distance = sg.create_distance('EuclideanDistance')
	distance.init(data, data)
	kmeans=sg.create_machine("KMeans", k=k, distance=distance, seed=1)

	#print("Running clustering...")
	kmeans.train()

	return kmeans.get("cluster_centers")

def assign_labels(data, centroids, ncenters):
	from shogun import MulticlassLabels
	import shogun as sg
	from numpy import arange

	labels = MulticlassLabels(arange(0.,ncenters))
	fea_centroids = sg.create_features(centroids)
	distance = sg.create_distance('EuclideanDistance')
	distance.init(fea_centroids, fea_centroids)
	knn = sg.create_machine("KNN", k=1, distance=distance)
	knn.train(fea_centroids, labels)
	return knn.apply(data)

def evaluation_clustering_simple (n_data=100, sqrt_num_blobs=4, distance=5):
	from shogun import ClusteringAccuracy, ClusteringMutualInformation
	from shogun import MulticlassLabels, GaussianBlobsDataGenerator

	# produce sone Gaussian blobs to cluster
	ncenters=sqrt_num_blobs**2
	stretch=1
	angle=1
	gen=GaussianBlobsDataGenerator(sqrt_num_blobs, distance, stretch, angle)
	gen.put("seed", 1)
	features=gen.get_streamed_features(n_data)
	X=features.get("feature_matrix")

	# compute approximate "ground truth" labels via taking the closest blob mean
	coords=array(range(0,sqrt_num_blobs*distance,distance))
	idx_0=[abs(coords -x).argmin() for x in X[0]]
	idx_1=[abs(coords -x).argmin() for x in X[1]]
	ground_truth=array([idx_0[i]*sqrt_num_blobs + idx_1[i] for i in range(n_data)], dtype="float64")

	#for label in unique(ground_truth):
	#	indices=ground_truth==label
	#	plot(X[0][indices], X[1][indices], 'o')
	#show()

	centroids = run_clustering(features, ncenters)
	gnd_hat = assign_labels(features, centroids, ncenters)
	gnd = MulticlassLabels(ground_truth)

	AccuracyEval = ClusteringAccuracy()
	AccuracyEval.best_map(gnd_hat, gnd)

	accuracy = AccuracyEval.evaluate(gnd_hat, gnd)
	# in this case we know that the clustering has to be very good
	#print(('Clustering accuracy = %.4f' % accuracy))
	assert(accuracy>0.8)

	MIEval = ClusteringMutualInformation()
	mutual_info = MIEval.evaluate(gnd_hat, gnd)
	#print(('Clustering mutual information = %.4f' % mutual_info))

	return gnd, accuracy, mutual_info

if __name__ == '__main__':
	print('Evaluation Clustering')
	evaluation_clustering_simple(*parameter_list[0])
