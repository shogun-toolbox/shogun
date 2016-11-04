=======
K-means
=======
:math:`K`-means clustering aims to partition :math:`n` observations into :math:`k\leq n` clusters (sets :math:`\mathbf{S}`),
in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster.

In other words, its objective is to minimize:

.. math::
   \argmin_\mathbf{S} \sum_{i=1}^{k}\sum_{\mathbf{x}\in S_k}\left \|\boldsymbol{x} - \boldsymbol{\mu}_i  \right \|^{2}

where :math:`\mathbf{μ}_i` is the mean of points in :math:`S_i`.

See Chapter 20 in :cite:`barber2012bayesian` for a detailed introduction.

-------
Example
-------
Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) as

.. sgexample:: kmeans.sg:create_features

In order to run :sgclass:`CKMeans`, we need to choose a distance, for example :sgclass:`CEuclideanDistance`, or other sub-classes of :sgclass:`CDistance`. The distance is initialized with the data we want to classify.

.. sgexample:: kmeans.sg:choose_distance

Once we have chosen a distance, we create an instance of the :sgclass:`CKMeans` classifier.
We explicitly set :math:`k`, the number of clusters we are expecting to have as 3 and pass it to :sgclass:`CKMeans`. In this example, we apply Lloyd's method for `k`-means clustering.

.. sgexample:: kmeans.sg:create_instance_lloyd

Then we train the model:

.. sgexample:: kmeans.sg:train_dataset

We can extract centers and radius of each cluster:

.. sgexample:: kmeans.sg:extract_centers_and_radius


:sgclass:`CKMeans` also supports mini batch :math:`k`-means clustering.
We can create an instance of :sgclass:`CKMeans` classifier with mini batch :math:`k`-means method by providing the batch size and iteration number.

.. sgexample:: kmeans.sg:create_instance_mb

Then train the model and extract the centers and radius information as mentioned above.

----------
References
----------
:wiki:`K-means_clustering`

:wiki:`Lloyd's_algorithm`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
