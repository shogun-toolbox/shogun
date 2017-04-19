================
Minkowski Metric
================

The Minkowski metric is one general class of metrics for a
:math:`\displaystyle R^{n}` feature space also referred as 
the :math:`\displaystyle L_{k}` norm.

:math:`d(\bf{x},\bf{x'}) = (\sum_{i=1}^{n} |\bf{x_{i}}- \ 
\bf{x'_{i}}|^{k})^{\frac{1}{k}} \quad x,x' \in R^{n}`

Note that the Minkowski distance tends to the Chebyshew distance for
increasing :math:`k`.
 
-------
Example
-------

We start by creating CDenseFeatures (here 64 bit floats aka RealFeatures) from files with training and test data.
.. sgexample:: minkowski_distance.sg:create_features

Then, we create an instance of MinkowskiMetric, passing it the training data.
.. sgexample:: minkowski_distance.sg:create_distance

Subsequently, we retrieve the distance for the training data. After that we initialize both training and test data and retrieve the distance of the test data.
.. sgexample:: minkowski_distance.sg:train_and_init

----------
References
----------
:wiki:`Distance`

.. bibliography:: ../../references.bib
    :filter: docname in docnames

