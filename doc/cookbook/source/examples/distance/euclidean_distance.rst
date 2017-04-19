==================
Euclidean distance
==================

The familiar Euclidean distance for real valued features computes
the square root of the sum of squared disparity between the
corresponding feature dimensions of two data points.

:math:`d({\bf x},{\bf x'})= \sqrt{\sum_{i=0}^{n}|{\bf x_i}-{\bf x'_i}|^2}`

This special case of Minkowski metric is invariant to an arbitrary
translation or rotation in feature space.

The Euclidean Squared distance does not take the square root:

:math:`d({\bf x},{\bf x'})= \sum_{i=0}^{n}|{\bf x_i}-{\bf x'_i}|^2`
 
Distance is computed as :
 
:math:`\sqrt{{\bf x}\cdot {\bf x} - 2{\bf x}\cdot {\bf x'} + {\bf x'}\cdot {\bf x'}}`
  
Squared norms for left hand side and right hand side features can be precomputed.

WARNING : Make sure to reset squared norms using reset_squared_norms() when features
or feature matrix are changed.
 
-------
Example
-------

We start by creating CDenseFeatures (here 64 bit floats aka RealFeatures) from files with training and test data.
.. sgexample:: euclidean_distance.sg:create_features

Then, we create an instance of EuclideanDistance, passing it the training data.
.. sgexample:: euclidean_distance.sg:create_distance

Subsequently, we retrieve the distance for the training data. After that we initialize both training and test data and retrieve the distance of the test data.
.. sgexample:: euclidean_distance.sg:train_and_init

----------
References
----------
:wiki:`Distance#Distance_in_Euclidean_space`

.. bibliography:: ../../references.bib
    :filter: docname in docnames

