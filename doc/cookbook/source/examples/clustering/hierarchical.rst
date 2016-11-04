=======================
Hierarchical Clustering
=======================

Hierarchical clustering is a method of cluster analysis which seeks to build a hierarchy of clusters.
We apply a "bottom up" approach: each observation starts in its own clister, and pairs of clusters are subsequently merged.

The merges are determined in a greedy manner.
We start by constructing a pairwise distance matrix. Then, the clusters of the pair with closest distance are merged iteratively.

-------
Example
-------

Imagine we have files with the training data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) as:

.. sgexample:: hierarchical.sg:create_features

In order to run :sgclass:`CHierarchical`, we need to choose a distance, for example :sgclass:`CEuclideanDistance`, or other sub-classes of :sgclass:`CDistance`. The distance is initialized with the data we want to classify.

.. sgexample:: hierarchical.sg:choose_distance

We then create an instance of the :sgclass:`CHierarchical` classifier by assigning the steps of merging we expect to have in the training.

.. sgexample:: hierarchical.sg:create_instance

We can extract the information of the two merged elements, as well as the distance between them in each merging step:

.. sgexample:: hierarchical.sg:extract_results

----------
References
----------
:wiki:`Hierarchical_clustering`
