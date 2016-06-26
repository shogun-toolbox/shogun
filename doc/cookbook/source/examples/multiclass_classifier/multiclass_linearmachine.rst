==========================
Multi-class Linear Machine
==========================

We extend the application of linear machines to multi-class datasets by constructing generic multiclass classifiers with ensembles of binary classifiers.

In this example, we show how to apply :sgclass:`CLibLinear` to multi-class cases with :sgclass:`CLinearMulticlassMachine`.

`See the linear SVM cookbook <http://shogun.ml/cookbook/latest/examples/binary_classifier/linear_svm.html>`_ for the infomration about :sgclass:`CLibLinear` binary classifier.

-------
Example
-------

Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`CMulticlassLabels` as

.. sgexample:: multiclass_linearmachine.sg:create_features

We use :sgclass:`CLibLinear` as base classifier and create an instance of :sgclass:`CLibLinear`.

.. sgexample:: multiclass_linearmachine.sg:create_classifier

In order to run :sgclass:`CLinearMulticlassMachine`, we need to specify an multi-class strategy from :sgclass:`CMulticlassOneVsRestStrategy` and :sgclass:`CMulticlassOneVsOneStrategy`.

.. sgexample:: multiclass_linearmachine.sg:choose_strategy

We create an instance of the :sgclass:`CLinearMulticlassMachine` classifier by passing it the strategy, dataset, binary classifer and the labels.

.. sgexample:: multiclass_linearmachine.sg:create_instance

Then we train and apply it to test data, which here gives :sgclass:`CMulticlassLabels`.

.. sgexample:: multiclass_linearmachine.sg:train_and_apply

We can evaluate test performance via e.g. :sgclass:`CMulticlassAccuracy`.

.. sgexample:: multiclass_linearmachine.sg:evaluate_accuracy

----------
References
----------

:wiki:`Multiclass_classification`
