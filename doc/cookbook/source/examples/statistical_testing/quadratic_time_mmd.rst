==================
Quadratic Time MMD
==================

The quadratic time MMD implements a nonparametric statistical hypothesis test to reject the null hypothesis that to distributions :math:`p` and :math:`q`, only observed via :math:`n` and :math:`m` samples respectively, are the same, i.e. :math:`H_0:p=q`.

The (biased) test statistic is given by

.. math::

  \frac{1}{nm}\sum_{i=1}^n\sum_{j=1}^m k(x_i,x_i) + k(x_j, x_j) - 2k(x_i,x_j).
  

See :cite:`gretton2012kernel` for a detailed introduction.

-------
Example
-------

Imagine we have samples from :math:`p` and :math:`q`, here in the form of CDenseFeatures (here 64 bit floats aka RealFeatures).

.. sgexample:: quadratic_time_mmd.sg:create_features

We create an instance of :sgclass:`CQuadraticTimeMMD`, passing it data the kernel.

.. sgexample:: quadratic_time_mmd.sg:create_instance

We can select multiple ways to compute the test statistic, see :sgclass:`CQuadraticTimeMMD` for details. 
The biased statistic is computed as

.. sgexample:: quadratic_time_mmd.sg:estimate_mmd

There are multiple ways to perform the actual hypothesis test, see :sgclass:`CQuadraticTimeMMD` for details. The permutation version simulates from :math:`H_0` via repeatedly permuting the samples from :math:`p` and :math:`q`. We can perform the test via computing a test threshold for a given :math:`\alpha`, or by directly computing a p-value.

.. sgexample:: quadratic_time_mmd.sg:perform_test

----------------
Multiple kernels
----------------

It is possible to perform all operations (computing statistics, performing test, etc) for multiple kernels at once, via the :sgclass:`CMultiKernelQuadraticTimeMMD` interface.

.. sgexample:: quadratic_time_mmd.sg:multi_kernel

Note that the results are now a vector with one entry per kernel.
Also note that the kernels for single and multiple are kept separately.

---------------
Kernel learning
---------------

There are various options to learn a kernel.
All options allow to learn a single kernel among a number of provided baseline kernels.
Furthermore, some of these criterions can be used to learn the coefficients of a convex combination of baseline kernels.

There are different strategies to learn the kernel, see :sgclass:`CKernelSelectionStrategy`.

We specify the desired baseline kernels to consider. Note the kernel above is not considered in the selection.

.. sgexample:: quadratic_time_mmd.sg:add_kernels

IMPORTANT: when learning the kernel for statistical testing, this needs to be done on different data than being used for performing the actual test.
One way to accomplish this is to manually provide a different set of features for testing.
In Shogun, it is also possible to automatically split the provided data by specifying the ratio between train and test data, via enabling the train-test mode.

.. sgexample:: quadratic_time_mmd.sg:enable_train_test_mode

A ratio of 1 means the data is split into half during learning the kernel, and subsequent tests are performed on the second half.

We learn the kernel and extract the result, again see :sgclass:`CKernelSelectionStrategy` more available strategies.
Note that the kernel of the mmd itself is replaced.
If all kernels have the same type, we can convert the result into that type, for example to extract its parameters.

.. sgexample:: quadratic_time_mmd.sg:select_kernel_single

Note that in order to extract particular kernel parameters, we need to cast the kernel to its actual type.

Similarly, a convex combination of kernels, in the form of :sgclass:`CCombinedKernel` can be learned and extracted as

.. sgexample:: quadratic_time_mmd.sg:select_kernel_combined

We can perform the test on the last learnt kernel.
Since we enabled the train-test mode, this automatically is done on the held out test data.

.. sgexample:: quadratic_time_mmd.sg:perform_test_optimized

----------
References
----------
.. bibliography:: ../../references.bib
    :filter: docname in docnames

:wiki:`Statistical_hypothesis_testing`
