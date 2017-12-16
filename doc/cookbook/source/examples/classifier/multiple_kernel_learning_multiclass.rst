======================================================
Multiple Kernel Learning for Multi-class Classification
======================================================

Multiple kernel learning (MKL) is based on convex combinations of arbitrary kernels over potentially different domains.

.. math::

    {\bf k}(x_i,x_j)=\sum_{i=1}^{K} \beta_k {\bf k}_i(x_i, x_j)

where :math:`\beta_k > 0`, :math:`\sum_{k=1}^{K} \beta_k = 1`, :math:`K` is the number of sub-kernels, :math:`\bf{k}` is a combined kernel, :math:`{\bf k}_i` is an individual kernel and :math:`{x_i}_i` are the training data.

Classification is done by using :sgclass:`CMulticlassSVM`.

-------
Example
-------

Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`CMulticlassLabels` as

.. sgexample:: multiple_kernel_learning_multiclass.sg:create_features

Then we create indvidual kernels like :sgclass:`CPolyKernel`, :sgclass:`CGaussianKernel` and :sgclass:`CLinearKernel` which will be later combined in one :sgclass:`CCombinedKernel`.

.. sgexample:: multiple_kernel_learning_multiclass.sg:create_kernel

We create an instance of :sgclass:`CCombinedKernel` and append the :sgclass:`CKernel` objects.

.. sgexample:: multiple_kernel_learning_multiclass.sg:create_combined_train

We create an object of :sgclass:`CMKLMulticlass`, set necessary parameters, provide the combined kernel and labels before training it.

.. sgexample:: multiple_kernel_learning_multiclass.sg:train_mkl

After training, we can extract :math:`\beta`.

.. sgexample:: multiple_kernel_learning_multiclass.sg:extract_weights

We set the updated kernel and predict :sgclass:`CMulticlassLabels` for test data.

.. sgexample:: multiple_kernel_learning_multiclass.sg:mkl_apply

Finally, we can evaluate the :sgclass:`CMulticlassAccuracy`.

.. sgexample:: multiple_kernel_learning_multiclass.sg:evaluate_accuracy

----------
References
----------
:wiki:`Multiple_kernel_learning`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
