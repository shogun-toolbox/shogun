============
Relaxed Tree
============

The relaxed tree algorithm, or relaxed hierarchy algorithm, solves multi-class classification problems by exploiting the relaxed hierarchy structure of the data,

At each node, a binary classifier separates the data into three groups.
Labels :math:`1` and :math:`−1` mark the positive and negative sample groups assigned by the classifier,
while the confusing class, labeled with :math:`0`, are ignored by the binary classifier (what “relaxed” refers to).
The child of each node contains either group :math:`0` and :math:`1`, or group :math:`0` and :math:`−1`.

See :cite:`gao2011discriminative` for a detailed introduction.

-------
Example
-------

Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`CMulticlassLabels` as

.. sgexample:: relaxed_tree.sg:create_features

In order to run :sgclass:`CRelaxedTree`, we need to set the machine for confusion matrix and choose the kernel.

.. sgexample:: relaxed_tree.sg:set_parameters

We create an instance of the :sgclass:`CRelaxedTree` classifier, set the labels, and set the machine for confusion matrix and kernel.
We use confusion matrix to estimate the initial partition of the dataset and the kernel to train the model.

.. sgexample:: relaxed_tree.sg:create_instance

Then we train and apply it to test data, which here gives :sgclass:`CMulticlassLabels`.

.. sgexample:: relaxed_tree.sg:train_and_apply

We can evaluate test performance via e.g. :sgclass:`CMulticlassAccuracy`.

.. sgexample:: relaxed_tree.sg:evaluate_accuracy

----------
References
----------

.. bibliography:: ../../references.bib
    :filter: docname in docnames
