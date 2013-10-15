#!/usr/bin/env python
def get_dataset():
	from os.path import exists
	filename = "../data/optdigits.tes"
	if exists(filename):
		return open(filename)
	else:
		#print("Retrieving data...")
		try:
			from urllib2 import urlopen
		except ImportError:
			from urllib.request import urlopen
		return urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes")

def prepare_data():
	from numpy import loadtxt
	stream = get_dataset()
	#print("Loading data...")
	data = loadtxt(stream, delimiter=',')
	fea = data[:, :-1]
	gnd = data[:, -1]
	return (fea.T, gnd)

(fea, gnd_raw) = prepare_data()
parameter_list = [[fea, gnd_raw, 10]]


def run_clustering(data, k):
	from modshogun import KMeans
	from modshogun import EuclideanDistance
	from modshogun import RealFeatures

	fea = RealFeatures(data)
	distance = EuclideanDistance(fea, fea)
	kmeans=KMeans(k, distance)

	#print("Running clustering...")
	kmeans.train()

	return kmeans.get_cluster_centers()

def assign_labels(data, centroids, ncenters):
	from modshogun import EuclideanDistance
	from modshogun import RealFeatures, MulticlassLabels
	from modshogun import KNN
	from numpy import arange

	labels = MulticlassLabels(arange(0.,ncenters))
	fea = RealFeatures(data)
	fea_centroids = RealFeatures(centroids)
	distance = EuclideanDistance(fea_centroids, fea_centroids)
	knn = KNN(1, distance, labels)
	knn.train()
	return knn.apply(fea)

def evaluation_clustering (features=fea, ground_truth=gnd_raw, ncenters=10):
	from modshogun import ClusteringAccuracy, ClusteringMutualInformation
	from modshogun import MulticlassLabels
	from modshogun import Math

	# reproducable results
	Math.init_random(1)

	centroids = run_clustering(features, ncenters)
	gnd_hat = assign_labels(features, centroids, ncenters)
	gnd = MulticlassLabels(ground_truth)

	AccuracyEval = ClusteringAccuracy()
	AccuracyEval.best_map(gnd_hat, gnd)

	accuracy = AccuracyEval.evaluate(gnd_hat, gnd)
	#print(('Clustering accuracy = %.4f' % accuracy))

	MIEval = ClusteringMutualInformation()
	mutual_info = MIEval.evaluate(gnd_hat, gnd)
	#print(('Clustering mutual information = %.4f' % mutual_info))

	# TODO mutual information does not work with serialization
	#return gnd, gnd_hat, accuracy, MIEval, mutual_info
	return gnd, gnd_hat, accuracy

if __name__ == '__main__':
	print('Evaluation Clustering')
	evaluation_clustering(*parameter_list[0])
