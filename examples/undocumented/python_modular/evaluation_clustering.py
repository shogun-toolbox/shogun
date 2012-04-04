##!/usr/bin/env python
# Example on how to evaluate the clustering performance (given ground-truth)

from shogun.Distance import EuclidianDistance
from shogun.Features import RealFeatures
from shogun.Features import Labels
from shogun.Evaluation import ClusteringAccuracy
from shogun.Evaluation import ClusteringMutualInformation

def get_dataset():
    from os.path import exists
    from urllib2 import urlopen

    filename = "../data/optdigits.tes"
    if exists(filename):
        return open(filename)
    else:
        print("Retrieving data...")
        return urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes")

def prepare_data():
    from numpy import loadtxt

    stream = get_dataset()
    print("Loading data...")
    data = loadtxt(stream, delimiter=',')
    fea = data[:, :-1]
    gnd = data[:, -1]
    return (fea.T, gnd)

def run_clustering(data, k):
    from shogun.Clustering import KMeans
    from shogun.Mathematics import Math_init_random

    Math_init_random(42)
    fea = RealFeatures(data)
    distance = EuclidianDistance(fea, fea)
    kmeans=KMeans(k, distance)

    print("Running clustering...")
    kmeans.train()

    return kmeans.get_cluster_centers()

def assign_labels(data, centroids):
    from shogun.Classifier import KNN
    from numpy import arange

    labels = Labels(arange(1.,11.))
    fea = RealFeatures(data)
    fea_centroids = RealFeatures(centroids)
    distance = EuclidianDistance(fea_centroids, fea_centroids)
    knn = KNN(1, distance, labels)
    knn.train()
    return knn.apply(fea)

if __name__ == '__main__':
    (fea, gnd_raw) = prepare_data()
    centroids = run_clustering(fea, 10)
    gnd_hat = assign_labels(fea, centroids)
    gnd = Labels(gnd_raw)

    AccuracyEval = ClusteringAccuracy()
    AccuracyEval.best_map(gnd_hat, gnd)

    with open('/tmp/foo.txt', 'w') as ous:
        for i in range(gnd_hat.get_num_labels()):
            ous.write('%d ' % gnd_hat.get_int_label(i))
        ous.write('\n')
        for i in range(gnd.get_num_labels()):
            ous.write('%d ' % gnd.get_int_label(i))
        ous.write('\n')

    accuracy = AccuracyEval.evaluate(gnd_hat, gnd)
    print('Clustering accuracy = %.4f' % accuracy)

    MIEval = ClusteringMutualInformation()
    mutual_info = MIEval.evaluate(gnd_hat, gnd)
    print('Clustering mutual information = %.4f' % mutual_info)

