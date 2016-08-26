=================
Canberra distance
=================

The Canberra distance sums up the dissimilarity (ratios) between feature
dimensions of two data points.

:math:`d(\mathbf{x},\mathbf{x'}) = \sum_{i=1}^{n} \
\frac{|\mathbf{x_{i}-\mathbf{x'_{i}}}|}{|\bf{x_{i}}|+|\bf{x'_{i}}|} \
\quad \bf{x},\bf{x'} \in R^{n}`

A summation element has range [0,1]. Note that :math:`d(x,0)=d(0,x')=n`
and :math:`d(0,0)=0`.
 
-------
Example
-------

We start by creating CDenseFeatures (here 64 bit floats aka RealFeatures) from files with training and test data.

.. sgexample:: canberra_distance.sg:create_features

Then, we create an instance of CanberraMetric, passing it the training data.

.. sgexample:: canberra_distance.sg:create_distance

Subsequently, we retrieve the distance for the training data. After that we initialize both training and test data and retrieve the distance of the test data.

.. sgexample:: canberra_distance.sg:train_and_init

----------
References
----------
:wiki:`Canberra_distance`

.. bibliography:: ../../references.bib
    :filter: docname in docnames

