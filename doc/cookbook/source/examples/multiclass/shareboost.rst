==========
ShareBoost
==========

The shareBoost algorithm learns a multiclass predictor from a subset of shared features of the samples with forward greedy selection approach.

See :cite:`shalev2011shareboost` for a detailed introduction.

-------
Example
-------
Imagine we have files with training and test data. We create DenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`MulticlassLabels` as

.. sgexample:: shareboost.sg:create_features

We create an instance of the :sgclass:`CShareBoost` classifier by setting the number of features expected to be used for learning.

.. sgexample:: shareboost.sg:create_instance

Then we train and apply it to test data, which gives :sgclass:`MulticlassLabels`.

.. sgexample:: shareboost.sg:train_and_apply

We can evaluate test performance via e.g. :sgclass:`MulticlassAccuracy`.

.. sgexample:: shareboost.sg:evaluate_accuracy

----------
References
----------

.. bibliography:: ../../references.bib
    :filter: docname in docnames
