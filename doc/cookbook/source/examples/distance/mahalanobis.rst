====================
Mahalanobis Distance
====================

The Mahalanobis distance for real valued features computes the distance between a feature vector and a distribution of features characterized by its mean and covariance.

.. math::

    \sqrt{ ( x_{i} -  \mu  )^\top   S^{-1} ( x_{i} -  \mu  )} 

-------
Example
-------

Imagine we have files with data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) as

.. sgexample:: mahalanobis.sg:create_features

We create an instance of :sgclass:`CMahalanobisDistance` by passing it :sgclass:`CDenseFeatures`.

.. sgexample:: mahalanobis.sg:create_instance

The distance matrix can be extracted as follows:

.. sgexample:: mahalanobis.sg:extract_distance

We can use the same instance with new :sgclass:`CDenseFeatures` to compute asymmetrical distance as follows:

.. sgexample:: mahalanobis.sg:refresh_distance

----------
References
----------
:wiki:`Mahalanobis_distance`

.. bibliography:: ../../references.bib
    :filter: docname in docnames