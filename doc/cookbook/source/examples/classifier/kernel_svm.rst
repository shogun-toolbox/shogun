=============================
Kernel Support Vector Machine
=============================

Kernel Support Vector Machine is a binary classifier which finds a data-separating hyperplane in a Hilbert space induced by a positive definite kernel. The hyperplane is chosen to maximize the margins between the two classes. The loss function that needs to be minimized is:

.. math::

    \max_{\bf \alpha} \sum_{i=0}^{N-1} \alpha_i - \sum_{i=0}^{N-1}\sum_{j=0}^{N-1} \alpha_i y_i \alpha_j y_j  k({\bf x}_i, {\bf x}_j)

subject to:

.. math::

    0 \leq \alpha_i \leq C, \sum_{i=0}^{N-1} \alpha_i y_i = 0

where :math:`N` is the number of training samples, :math:`{\bf x}_i` are training samples, :math:`k` is a kernel, :math:`\alpha_i` are the weights, :math:`y_i` is the corresponding label where :math:`y_i \in \{-1,+1\}` and :math:`C` is a pre-specified regularization parameter.

This example uses LibSVM :cite:`chang2011libsvm` as backend, Shogun has many more SVM implementations, see :sgclass:`CSVM`.

See :cite:`scholkopf2002learning` and Chapter 6 in :cite:`cristianini2000introduction` for a detailed introduction.

-------
Example
-------

Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`CBinaryLabels` as

.. sgexample:: kernel_svm.sg:create_features

In order to run :sgclass:`CLibSVM`, we need to initialize a kernel like :sgclass:`CGaussianKernel` with training features and some parameters like :math:`C` and epsilon i.e. residual convergence parameter which is optional.

.. sgexample:: kernel_svm.sg:set_parameters

We create an instance of the :sgclass:`CLibSVM` classifier by passing it regularization coefficient, kernel and labels.

.. sgexample:: kernel_svm.sg:create_instance

Then we train and apply it to test data, which here gives :sgclass:`CBinaryLabels`.

.. sgexample:: kernel_svm.sg:train_and_apply

We can extract :math:`\alpha` and :math:`b`.

.. sgexample:: kernel_svm.sg:extract_weights_bias

Finally, we can evaluate test performance via e.g. :sgclass:`CAccuracyMeasure`.

.. sgexample:: kernel_svm.sg:evaluate_accuracy

----------
References
----------
:wiki:`Support_vector_machine`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
