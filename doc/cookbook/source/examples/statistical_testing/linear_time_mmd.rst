===============
Linear Time MMD
===============

The linear time MMD implements a nonparametric statistical hypothesis test to reject the null hypothesis that to distributions :math:`p` and :math:`q`, each only observed via :math:`n` samples, are the same, i.e. :math:`H_0:p=q`.

The (unbiased) statistic is given by

.. math::

  \frac{2}{n}\sum_{i=1}^n k(x_{2i},x_{2i}) + k(x_{2i+1}, x_{2i+1}) - 2k(x_{2i},x_{2i+1}).

See :cite:`gretton2012kernel` for a detailed introduction.

-------
Example
-------

Imagine we have samples from :math:`p` and :math:`q`.
As the linear time MMD is a streaming statistic, we need to pass it `CStreamingFeatures`.
Here, we use synthetic data generators, but it is possible to construct `CStreamingFeatures` from (large) files.

.. sgexample:: linear_time_mmd.sg:create_features

We create an instance of :sgclass:`CLinearTimeMMD`, passing it data and the kernel to use,

.. sgexample:: linear_time_mmd.sg:create_instance

An important parameter for controlling the efficiency of the linear time MMD is block size of the number of samples that is processed at once. As a guideline, set as large as memory allows.

.. sgexample::linear_time_mmd.sg:set_burst

Computing the statistic is done as

.. sgexample::linear_time_mmd.sg:estimate_mmd

We can perform the hypothesis test via computing the rejection threshold

.. sgexample::linear_time_mmd.sg:perform_test_threshold

Alternatively, we can compute the p-value for the above value of the statistic

.. sgexample::linear_time_mmd.sg:perform_test_p_value

----------
References
----------
.. bibliography:: ../../references.bib
    :filter: docname in docnames
