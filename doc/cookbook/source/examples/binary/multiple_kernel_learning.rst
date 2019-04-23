========================
Multiple Kernel Learning
========================

Multiple kernel learning (MKL) is based on convex combinations of arbitrary kernels over potentially different domains.

.. math::

    {\bf k}(x_i,x_j)=\sum_{i=1}^{K} \beta_k {\bf k}_i(x_i, x_j)

where :math:`\beta_k > 0`, :math:`\sum_{k=1}^{K} \beta_k = 1`, :math:`K` is the number of sub-kernels, :math:`\bf{k}` is a combined kernel, :math:`{\bf k}_i` is an individual kernel and :math:`{x_i}_i` are the training data.

Classification is done by using Support Vector Machines (SVM). See :doc:`linear_support_vector_machine` for more details. Optimal :math:`\alpha` and :math:`b` for SVM and :math:`\beta` are determined via training.

See :cite:`sonnenburg2006large` for more details.

-------
Example
-------

Imagine we have files with training and test data. We create DenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`BinaryLabels` as

.. sgexample:: multiple_kernel_learning.sg:create_features

Then we create indvidual kernels like :sgclass:`CPolyKernel` and :sgclass:`GaussianKernel` which will be later combined in one :sgclass:`CombinedKernel`.

.. sgexample:: multiple_kernel_learning.sg:create_kernel

We create an instance of :sgclass:`CombinedKernel` and append the :sgclass:`Kernel` objects.

.. sgexample:: multiple_kernel_learning.sg:create_combined_train

We create an object of :sgclass:`MKLClassification`, provide the combined kernel and labels before training it.

.. sgexample:: multiple_kernel_learning.sg:train_mkl

After training, we can extract :math:`\beta`, SVM coefficients :math:`\alpha` and :math:`b`.

.. sgexample:: multiple_kernel_learning.sg:extract_weights

We update the :sgclass:`CombinedKernel` object for testing data.

.. sgexample:: multiple_kernel_learning.sg:create_combined_test

We set the updated kernel and predict :sgclass:`BinaryLabels` for test data.

.. sgexample:: multiple_kernel_learning.sg:mkl_apply

Finally, we can evaluate test performance via e.g. :sgclass:`CAccuracyMeasure`.

.. sgexample:: multiple_kernel_learning.sg:evaluate_accuracy

----------
References
----------
:wiki:`Multiple_kernel_learning`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
