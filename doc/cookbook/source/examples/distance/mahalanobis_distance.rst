====================
Mahalanobis distance
====================

The Mahalanobis distance for real valued features computes the distance
between a feature vector and a distribution of features characterized by its
mean and covariance.

:math:`D = \sqrt{ (x_i - \mu)^T \Sigma^{-1} (x_i - \mu)  }`

The Mahalanobis Squared distance does not take the square root:

:math:`D = (x_i - \mu)^T \Sigma^{-1} (x_i - \mu)`

If use_mean is set to false (which it is by default) the distance is computed
as

:math:`D = \sqrt{ (x_i - x_i')^T \Sigma^{-1} (x_i - x_i')  }`

i.e., instead of the mean as reference two vector :math:`x_i` and :math:`x_i'`
are compared.

 
-------
Example
-------

We start by creating CDenseFeatures (here 64 bit floats aka RealFeatures) from files with training and test data.
.. sgexample:: mahalanobis_distance.sg:create_features

Then, we create an instance of MahalanobisDistance, passing it the training data.
.. sgexample:: mahalanobis_distance.sg:create_distance

Subsequently, we retrieve the distance for the training data. After that we initialize both training and test data and retrieve the distance of the test data.
.. sgexample:: mahalanobis_distance.sg:train_and_init

----------
References
----------
:wiki:`Mahalanobis distance`

.. bibliography:: ../../references.bib
    :filter: docname in docnames

