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

Load some training and test data.

.. sgexample:: knn.sg:load_data

In order to run KNN, we need to choose a distance, for example CEuclideanDistance, or other sub-classes of CDistance. The distance is initialized with the data we want to classify.

.. sgexample:: knn.sg:choose_distance

Once we have chosen a distance, we create an instance of the CKNN classifier, passing it training data and labels.

.. sgexample:: knn.sg:create_instance

Then we run the KNN algorithm and apply it to test data.

.. sgexample:: knn.sg:train_and_apply

----------
References
----------
`KNN on Wikipedia <https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>`_

.. bibliography:: ../../references.bib
