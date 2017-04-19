==================
Chebyshev distance
==================

The Chebyshev distance (:math:`L_{\infty}` norm) returns the maximum of
absolute feature dimension differences between two data points.

:math:`d(\bf{x},\bf{x'}) = max|\bf{x_{i}}-\bf{x'_{i}}| \quad x,x' \in R^{n}`
 
-------
Example
-------

We start by creating CDenseFeatures (here 64 bit floats aka RealFeatures) from files with training and test data.
.. sgexample:: chebyshev_distance.sg:create_features

Then, we create an instance of ChebyshewMetric, passing it the training data.
.. sgexample:: chebyshev_distance.sg:create_distance

Subsequently, we retrieve the distance for the training data. After that we initialize both training and test data and retrieve the distance of the test data.
.. sgexample:: chebyshev_distance.sg:train_and_init

----------
References
----------
:wiki:`Chebyshev_distance`

.. bibliography:: ../../references.bib
    :filter: docname in docnames

