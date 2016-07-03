============================
Linear Discriminant Analysis
============================

This cookbook page introduces the application of
`linear discriminant analysis <http://shogun.ml/cookbook/latest/examples/binary_classifier/lda.html>`_
to multi-class classifications.

-------
Example
-------

Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`CMulticlassLabels` as

.. sgexample:: linear_discriminant_analysis.sg:create_features

We create an instance of the :sgclass:`CMCLDA` classifier with feature matrix and label list.
:sgclass:`CMCLDA` also has two default parameters, to set tolerance used in training and mark whether to store the within class covariances.

.. sgexample:: linear_discriminant_analysis.sg:create_instance

Then we train and apply it to the test data, which here gives :sgclass:`CMulticlassLabels`.

.. sgexample:: linear_discriminant_analysis.sg:train_and_apply

We can extract the mean vector of one class.
If we enabled storing covariance when creating instances, we can also extract the covariance matrix:

.. sgexample:: linear_discriminant_analysis.sg:extract_mean_and_cov

We can evaluate test performance via e.g. :sgclass:`CMulticlassAccuracy`.

.. sgexample:: linear_discriminant_analysis.sg:evaluate_accuracy

----------
References
----------

:wiki:`Linear_discriminant_analysis`

:wiki:`Linear_discriminant_analysis#Multiclass_LDA`
