==================
Quadratic Time MMD
==================

The Quadratic time MMD implements a nonparametric statistical hypothesis test to reject the null hypothesis that to distributions :math:`p` and :math:`q`, only observed via :math:`n` and :math:`m` samples respectively, are the same:

.. math::

  H_0:p=q.

The (biased) test statistic is given by

.. math::

  \frac{1}{nm}\sum_{i=1}^n\sum_{j=1}^m k(x_i,x_i) + k(x_j, x_j) - 2k(x_i,x_j).
  

See :cite:`gretton2012kernel` for a detailed introduction.

-------
Example
-------

Imagine we have samples from :math:`p` and :math:`q`. We create CDenseFeatures (here 64 bit floats aka RealFeatures)as

.. sgexample:: quadratic_time_mmd.sg:create_features

We create an instance of :sgclass:`CQuadraticTimeMMD`, passing it data the kernel, and the test significance level :math:`\alpha`

.. sgexample:: quadratic_time_mmd.sg:create_instance

We can select multiple ways to compute the test statistic, see :sgclass:`CQuadraticTimeMMD` for details. Unbiased statistic

.. sgexample:: quadratic_time_mmd.sg:estimate_mmd_unbiased

Biased statistic

.. sgexample:: quadratic_time_mmd.sg:estimate_mmd_biased

There are multiple ways to perform the actual hypothesis test. The permutation version simulates from :math:`H_0` via repeatedly permuting the samples from :math:`p` and :math:`q`:

.. sgexample:: quadratic_time_mmd.sg:perform_test_permutation

The spectrum version simulates from :math:`H_0` via approximating the Eigenspectrum of the underlying kernel:

.. sgexample:: quadratic_time_mmd.sg:perform_test_spectrum

The Gamma version fit a Gamma cumulative distribution function to :math:`H_0` via moment matching:

.. sgexample:: quadratic_time_mmd.sg:perform_test_gamma

----------
References
----------
.. bibliography:: ../../references.bib
    :filter: docname in docnames

:wiki:`Statistical_hypothesis_testing`
