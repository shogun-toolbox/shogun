====================
K Nearest neighbours
====================

KNN classifies data according to the majority of labels in the nearest neighbourhood, according to some underlying distance function :math:`d(x,x')`.

For :math:`k=1`, the label for a test point :math:`x^*` is predicted to be the same as for its closest training point :math:`x_{k}`, i.e. :math:`y_{k}`, where

.. math::

   k=\argmin_j d(x^*, x_j).  
   
See Chapter 14 in :cite:`barber2012bayesian` for a detailed introduction.

-------
Example
-------

Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`CMulticlassLabels` as

.. sgexample:: knn.sg:create_features

In order to run :sgclass:`CKNN`, we need to choose a distance, for example :sgclass:`CEuclideanDistance`, or other sub-classes of :sgclass:`CDistance`. The distance is initialized with the data we want to classify.

.. sgexample:: knn.sg:choose_distance

Once we have chosen a distance, we create an instance of the :sgclass:`CKNN` classifier, passing it :math:`k`.

.. sgexample:: knn.sg:create_instance

Then we run the train KNN algorithm and apply it to test data, which here gives :sgclass:`CMulticlassLabels`.

.. sgexample:: knn.sg:train_and_apply

We can evaluate test performance via e.g. :sgclass:`CMulticlassAccuracy`.

.. sgexample:: knn.sg:evaluate_accuracy


----------
References
----------
:wiki:`K-nearest_neighbors_algorithm`

.. bibliography:: ../../references.bib
