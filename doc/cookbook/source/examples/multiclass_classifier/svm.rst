==================================
Multi-class Support Vector Machine
==================================

The multi-class support vector machine is a multi-class classifier which uses :sgclass:`CLibSVM` to do one vs one classification. See :doc:`../binary_classifier/kernel_svm` for more details.

-------
Example
-------

Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`CMulticlassLabels` as

.. sgexample:: svm.sg:create_features

In order to run :sgclass:`CMulticlassLibSVM`, we need to initialize a kernel like :sgclass:`CGaussianKernel` with training features and some parameters like :math:`C` and epsilon i.e. residual convergence parameter which is optional.

.. sgexample:: svm.sg:set_parameters

We create an instance of the :sgclass:`CMulticlassLibSVM` classifier by passing it regularization coefficient, kernel and labels.

.. sgexample:: svm.sg:create_instance

Then we train and apply it to test data, which here gives :sgclass:`CMulticlassLabels`.

.. sgexample:: svm.sg:train_and_apply

Finally, we can evaluate test performance via e.g. :sgclass:`CMulticlassAccuracy`.

.. sgexample:: svm.sg:evaluate_error

----------
References
----------
:wiki:`Multiclass_classification`

:wiki:`Support_vector_machine`

:doc:`kernel_svm`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
