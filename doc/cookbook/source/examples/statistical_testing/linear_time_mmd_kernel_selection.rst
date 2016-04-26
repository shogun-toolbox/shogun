==================================
Linear Time MMD (kernel selection)
==================================

For the linear time MMD, it is possible to learn the kernel that maximizes the test power.
That is, for a fixed type I error, say :math:`\alpha=0.05`, the type II error is minimized.
Maximising the type II error here is equivalent to picking a kernel :math:`k` that maximizes 

.. math::

  \argmax_k \frac{\text{MMD}_l}{\sigma_l},
  
where :math:`\text{MMD}_l` is the linear time MMD estimator and :math:`\sigma_l` is its standard deviation, both of which can be estimated in an on-line fashion.
  
This allows to select a single kernel among a number of provided baseline kernels. 
Furthermore, it is possible to learn the coefficients of a convex combination of :math:`d` baseline kernels :math:`\sum_{i=1}^d \lambda_i k_i` via solving a convex program of the form

.. math::

  \argmax_\lambda \lambda^\top Q \lambda \qquad \text{subject to } (\lambda^\top h)=1 \quad \lambda_i\geq 0
  
where :math:`h` is a vector of MMD statistics for each kernel and :math:`Q` is its empirical covariance.

See :cite:`gretton2012optimal` for details.

-------
Example
-------

Imagine we have (streamed) samples from :math:`p` and :math:`q`. 
Note that the data used to learn the kernel must be *different* from the data used for the test in order to ensure correct calibration, :cite:`gretton2012optimal` for details.

We create an instance of :sgclass:`CLinearTimeMMD`, passing it the training data.

.. sgexample:: linear_time_mmd_kernel_selection.sg:create_instance

We then specify the desired baseline kernels to consider.

.. sgexample:: linear_time_mmd_kernel_selection.sg:add_kernels

The single kernel that maximizes the test power can be learned and extracted using

.. sgexample:: linear_time_mmd_kernel_selection.sg:select_kernel_single

Note that in order to extract particular kernel parameters, we need to cast the kernel to its actual type. Similarly, a convex combination of kernels, in the form of :sgclass:`CCombinedKernel`, that maximizes the test power can be learned and extracted using

.. sgexample:: linear_time_mmd_kernel_selection.sg:select_kernel_combined

We can perform the test on the last learnt kernel (note again, this must be done on different data)

.. sgexample:: linear_time_mmd_kernel_selection.sg:perform_test

----------
References
----------
.. bibliography:: ../../references.bib
    :filter: docname in docnames
