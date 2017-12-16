==================
Manhattan distance
==================

The Manhattan distance (city block distance, :math:`L_{1}` norm, rectilinear
distance or taxi cab metric ) is a special case
of general Minkowski metric and computes the absolute differences
between the feature dimensions of two data points.

:math:`d(\bf{x},\bf{x'}) = \sum_{i=1}^{n} |\bf{x_{i}}-\bf{x'_{i}}| \ 
\quad \bf{x},\bf{x'} \in R^{n}`
 
-------
Example
-------

We start by creating CDenseFeatures (here 64 bit floats aka RealFeatures) from files with training and test data.
.. sgexample:: manhattan_distance.sg:create_features

Then, we create an instance of ManhattanMetric, passing it the training data.
.. sgexample:: manhattan_distance.sg:create_distance

Subsequently, we retrieve the distance for the training data. After that we initialize both training and test data and retrieve the distance of the test data.
.. sgexample:: manhattan_distance.sg:train_and_init

----------
References
----------
:wiki:`Manhattan_distance`

.. bibliography:: ../../references.bib
    :filter: docname in docnames

