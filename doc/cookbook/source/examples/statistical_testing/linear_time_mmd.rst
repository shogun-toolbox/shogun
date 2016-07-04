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
As the linear time MMD is a streaming statistic, we need to pass it :sgclass:`CStreamingFeatures`.
Here, we use synthetic data generators, but it is possible to construct :sgclass:`CStreamingFeatures` from (large) files.
We create an instance of :sgclass:`CLinearTimeMMD`, passing it data and the kernel to use,

.. sgexample:: linear_time_mmd.sg:create_instance

An important parameter for controlling the efficiency of the linear time MMD is block size of the number of samples that is processed at once. As a guideline, set as large as memory allows.

.. sgexample::linear_time_mmd.sg:set_burst

Computing the statistic is done as

.. sgexample::linear_time_mmd.sg:estimate_mmd

We can perform the hypothesis test via computing a test threshold for a given :math:`\alpha`, or by directly computing a p-value.

.. sgexample::linear_time_mmd.sg:perform_test_threshold

----------------
Kernel selection
----------------

There are various options to learn a kernel.
All options allow to learn a single kernel among a number of provided baseline kernels.
Furthermore, some of these criterions can be used to learn the coefficients of a convex combination of baseline kernels.

There are different strategies to learn the kernel, see :sgclass:`CKernelSelectionStrategy`.

We specify the desired baseline kernels to consider. Note the kernel above is not considered in the selection.

.. sgexample:: linear_time_mmd.sg:add_kernels

IMPORTANT: when learning the kernel for statistical testing, this needs to be done on different data than being used for performing the actual test.
One way to accomplish this is to manually provide a different set of features for testing.
In Shogun, it is also possible to automatically split the provided data by specifying the ratio between train and test data, via enabling the train-test mode.

.. sgexample:: linear_time_mmd.sg:enable_train_test_mode

A ratio of 1 means the data is split into half during learning the kernel, and subsequent tests are performed on the second half.

We learn the kernel and extract the result, again see :sgclass:`CKernelSelectionStrategy` more available strategies. Note that the kernel of the mmd itself is replaced.
If all kernels have the same type, we can convert the result into that type, for example to extract its parameters.

.. sgexample:: linear_time_mmd.sg:select_kernel_single

Note that in order to extract particular kernel parameters, we need to cast the kernel to its actual type.

Similarly, a convex combination of kernels, in the form of :sgclass:`CCombinedKernel` can be learned and extracted as

.. sgexample:: linear_time_mmd.sg:select_kernel_combined

We can perform the test on the last learnt kernel.
Since we enabled the train-test mode, this automatically is done on the held out test data.

.. sgexample:: linear_time_mmd.sg:perform_test

----------
References
----------
.. bibliography:: ../../references.bib
    :filter: docname in docnames
