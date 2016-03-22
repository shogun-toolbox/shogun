==================
Quadratic Time MMD
==================

The unbiased statistic is given by

.. math::

  \frac{1}{n(n-1)}\sum_{i=1}^n\sum_{j=1}^n k(x_i,x_i) + k(x_j, x_j) - 2k(x_i,x_j).
  

See :cite:`gretton2012kernel` for a detailed introduction.

-------
Example
-------

Imagine we have samples from :math:`p` and :math:`p`. We create CDenseFeatures (here 64 bit floats aka RealFeatures)as

.. sgexample:: quadratic_time_mmd.sg:create_features

We create an instance of :sgclass:`CQuadraticTimeMMD`, passing it data and the kernel to use, a :sgclass:`CGaussianKernel` here.

.. sgexample:: quadratic_time_mmd.sg:create_instance

Computing the statistic is done as

.. sgexample:: quadratic_time_mmd.sg:estimate_mmd

We can perform the hypothesis test as

.. sgexample:: quadratic_time_mmd.sg:perform_test

----------
References
----------
.. bibliography:: ../../references.bib
    :filter: docname in docnames
