===============
Linear Time MMD
===============

The unbiased statistic is given by

.. math::

  \frac{2}{n}\sum_{i=1}^n k(x_{2i},x_{2i}) + k(x_{2i+1}, x_{2i+1}) - 2k(x_{2i},x_{2i+1}).
  

See :cite:`gretton2012kernel` for a detailed introduction.

-------
Example
-------

Imagine we have samples from :math:`p` and :math:`p`. We create CDenseFeatures (here 64 bit floats aka RealFeatures)as

.. sgexample:: linear_time_mmd.sg:create_features

We create an instance of :sgclass:`CLinearTimeMMD`, passing it data and the kernel to use, a :sgclass:`CGaussianKernel` here.

.. sgexample:: linear_time_mmd.sg:create_instance

Computing the statistic is done as

.. sgexample:: linear_time_mmd.sg:estimate_mmd

We can perform the hypothesis test as

.. sgexample:: linear_time_mmd.sg:perform_test

----------
References
----------
.. bibliography:: ../../references.bib
    :filter: docname in docnames
