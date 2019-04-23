==========================
Multi-class Linear Machine
==========================

We extend the application of linear machines to multi-class datasets by constructing generic multiclass classifiers with ensembles of binary classifiers.

In this example, we show how to apply :sgclass:`LibLinear` to multi-class cases with :sgclass:`LinearMulticlassMachine`.

`See the linear SVM cookbook <http://shogun.ml/cookbook/latest/examples/binary/linear_support_vector_machine.html>`_ for the infomration about :sgclass:`LibLinear` binary classifier.

-------
Example
-------

Imagine we have files with training and test data. We create DenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`MulticlassLabels` as

.. sgexample:: linear.sg:create_features

We use :sgclass:`LibLinear` as base classifier and create an instance of :sgclass:`LibLinear`.

.. sgexample:: linear.sg:create_classifier

In order to run :sgclass:`LinearMulticlassMachine`, we need to specify an multi-class strategy from :sgclass:`MulticlassOneVsRestStrategy` and :sgclass:`MulticlassOneVsOneStrategy`.

.. sgexample:: linear.sg:choose_strategy

We create an instance of the :sgclass:`LinearMulticlassMachine` classifier by passing it the strategy, dataset, binary classifer and the labels.

.. sgexample:: linear.sg:create_instance

Then we train and apply it to test data, which here gives :sgclass:`MulticlassLabels`.

.. sgexample:: linear.sg:train_and_apply

We can evaluate test performance via e.g. :sgclass:`MulticlassAccuracy`.

.. sgexample:: linear.sg:evaluate_accuracy

----------
References
----------

:wiki:`Multiclass_classification`
