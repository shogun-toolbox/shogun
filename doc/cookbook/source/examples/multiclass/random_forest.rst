=============
Random Forest
=============

A Random Forest is an ensemble learning method which implements multiple decision trees during training. It predicts by using a combination rule on the outputs of individual decision trees.

See :cite:`Breiman2001` for a detailed introduction.

-------
Example
-------

DenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`MulticlassLabels` are created from training and test data file

.. sgexample:: random_forest.sg:create_features

Combination rules to be used for prediction are derived form the :sgclass:`CombinationRule` class. Here we create a :sgclass:`MajorityVote` class to be used as a combination rule.

.. sgexample:: random_forest.sg:create_combination_rule

Next an instance of :sgclass:`CRandomForest` is created. The parameters provided are the number of attributes to be chosen randomly to select from and the number of trees.

.. sgexample:: random_forest.sg:create_instance

Then we run the train random forest and apply it to test data, which here gives :sgclass:`MulticlassLabels`.

.. sgexample:: random_forest.sg:train_and_apply

We can evaluate test performance via e.g. :sgclass:`MulticlassAccuracy` as well as get the "out of bag error".

.. sgexample:: random_forest.sg:evaluate_accuracy

----------
References
----------
:wiki:`Random_forest`

:wiki:`Out-of-bag_error`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
