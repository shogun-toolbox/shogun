===============================
Large Margin Nearest Neighbours
===============================

Large margin nearest neighbours is a metric learning algorithm.  It learns a metric that can be used with the :doc:`knn` algorithm.

The Mahalanobis distance metric which is an instance of :sgclass:`CCustomMahalanobisDistance` is obtained as a result.

See :cite:`weinberger2009distance` for a detailed introduction.

-------
Example
-------

Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`CMulticlassLabels` as

.. sgexample:: large_margin_nearest_neighbours.sg:create_features

We create an instance of :sgclass:`CLMNN` and provide number of nearest neighbours as parameters.

.. sgexample:: large_margin_nearest_neighbours.sg:create_instance

Next we train the LMNN algorithm and get the learned metric.

.. sgexample:: large_margin_nearest_neighbours.sg:train_metric

Then we train the :sgclass:`CKNN` algorithm using the learned metric and apply it to test data, which here gives :sgclass:`CMulticlassLabels`.

.. sgexample:: large_margin_nearest_neighbours.sg:train_and_apply

We can evaluate test performance via e.g. :sgclass:`CMulticlassAccuracy`.

.. sgexample:: large_margin_nearest_neighbours.sg:evaluate_accuracy


----------
References
----------
:wiki:`Large_margin_nearest_neighbor`

.. bibliography:: ../../references.bib
    :filter: docname in docnames

