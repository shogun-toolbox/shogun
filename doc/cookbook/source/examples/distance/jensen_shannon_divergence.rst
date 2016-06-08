=========================
Jensen–Shannon divergence
=========================

The Jensen-Shannon divergence (often reffered to as Jensen-Shannon distance) measures
the similarity between two data points which is based on the Kullback-Leibler divergence.

:math:`d(\bf{x},\bf{x'}) = \sum_{i=0}^{n} x_{i} log\frac{x_{i}}{0.5(x_{i}+x'_{i})} \ 
+ x'_{i} log\frac{x'_{i}}{0.5(x_{i}+x'_{i})}`
 
-------
Example
-------

We start by creating CDenseFeatures (here 64 bit floats aka RealFeatures) from files with training and test data.
.. sgexample:: jensen_shannon_divergence.sg:create_features

Then, we create an instance of JensenMetric, passing it the training data.
.. sgexample:: jensen_shannon_divergence.sg:create_distance

Subsequently, we retrieve the distance for the training data. After that we initialize both training and test data and retrieve the distance of the test data.
.. sgexample:: jensen_shannon_divergence.sg:train_and_init

----------
References
----------
:wiki:`Jensen-Shannon_divergence`
:wiki:`Kullback-Leibler_divergence`

.. bibliography:: ../../references.bib
    :filter: docname in docnames

