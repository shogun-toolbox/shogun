==================================
Classification And Regression Tree
==================================

Decision tree learning uses a decision tree as a predictive model which maps observations about an item to conclusions about the item's target value.

Decision trees are mostly used as the following two types:

- Classification tree, where the predicted outcome is the class to which the data belongs.
- Regression tree, where predicted outcome can be considered a real number.

Classification And Regression Tree (CART) algorithm is an umbrella method that can be applied to generate both classification tree and regression tree.

In this example, we showed how to apply CART algorithm to multi-class dataset and predict the labels with classification tree.

-------
Example
-------

Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`CMulticlassLabels` as

.. sgexample:: cartree.sg:create_features

We set the type of each predictive attribute (true for nominal, false for ordinal/continuous)

.. sgexample:: cartree.sg:set_attribute_types

We create an instance of the :sgclass:`CCARTree` classifier by passting it the attribute types and the tree type.
We can also set the number of subsets used in cross-valiation and whether to use cross-validation pruning.

.. sgexample:: cartree.sg:create_instance

Then we train and apply it to test data, which here gives :sgclass:`CMulticlassLabels`.

.. sgexample:: cartree.sg:train_and_apply

We can evaluate test performance via e.g. :sgclass:`CMulticlassAccuracy`.

.. sgexample:: cartree.sg:evaluate_accuracy

----------
References
----------

:wiki:`Decision_tree_learning`

:wiki:`Predictive_analytics#Classification_and_regression_trees_.28CART.29`
