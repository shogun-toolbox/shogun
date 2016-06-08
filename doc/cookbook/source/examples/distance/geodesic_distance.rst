=========================================
Geodesic distance (Great circle distance)
=========================================

The Geodesic distance (Great circle distance) computes the shortest path
between two data points on a sphere (the radius is set to one for the
evaluation).

:math:`d(\bf{x},\bf{x'}) = arccos\sum_{i=1}^{n} \ 
\frac{\bf{x_{i}}\cdot\bf{x'_{i}}} {\sqrt{x_{i}x_{i} x'_{i}x'_{i}}}`
 
-------
Example
-------

We start by creating CDenseFeatures (here 64 bit floats aka RealFeatures) from files with training and test data.
.. sgexample:: geodesic_distance.sg:create_features

Then, we create an instance of GeodesicMetric, passing it the training data.
.. sgexample:: geodesic_distance.sg:create_distance

Subsequently, we retrieve the distance for the training data. After that we initialize both training and test data and retrieve the distance of the test data.
.. sgexample:: geodesic_distance.sg:train_and_init

----------
References
----------
:wiki:`Great_circle_distance`

.. bibliography:: ../../references.bib
    :filter: docname in docnames

