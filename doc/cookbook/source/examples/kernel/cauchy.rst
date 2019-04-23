=============
Cauchy Kernel
=============

The cauchy kernel is derived from the Cauchy distribution :cite:`basak2008kernel` (Section 3).
It is a long-tailed kernel defined as

.. math::
  k(x,x') = \frac{1}{1+\frac{\| x-x' \|^2}{\sigma}}

, where :math:`|x-x'|` is the distance between two feature vectors given by an instance of :sgclass:`Distance`.

-------
Example
-------
Imagine we have files with data. 
We create :sgclass:`DenseFeatures` (here 64 bit floats aka RealFeatures) as

.. sgexample:: cauchy:create_features

We create an instance of :sgclass:`EuclideanDistance` and initialize with :sgclass:`DenseFeatures`.
Then we create an instance of :sgclass:`CCauchyKernel` with the distance.

.. sgexample:: cauchy:create_kernel

We initialize the kernel with :sgclass:`DenseFeatures`. 
The kernel matrix can be extracted as follows:

.. sgexample:: cauchy:kernel_matrix_train

We can use the same instance with new :sgclass:`DenseFeatures` to compute the kernel matrix between training features and testing features.

.. sgexample:: cauchy:kernel_matrix_test

----------
References
----------
.. bibliography:: ../../references.bib
    :filter: docname in docnames
