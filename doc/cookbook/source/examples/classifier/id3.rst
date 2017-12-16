====================
ID3 Classifier
====================

The ID3 algorithm is a decision tree based learning algorithm that can perform multi-class classification. It begins with the original set of attributes as the root node. On each iteration of the algorithm, it iterates through every unused attribute of the set and calculates the information gain of that attribute. It then selects the attribute which has the largest information gain value. The set is then split by the selected attribute. The algorithm continues to recurse on each subset until a stopping criteria is reached. 

-------
Example
-------

Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`CMulticlassLabels` as

.. sgexample:: id3.sg:create_features

We create an instance of the :sgclass:`CID3ClassifierTree` classifier and set labels.

.. sgexample:: id3.sg:create_instance

Then we run the train the ID3 Classifier algorithm and apply it to test data, which here gives :sgclass:`CMulticlassLabels`.

.. sgexample:: id3.sg:train_and_apply

We can evaluate test performance via e.g. :sgclass:`CMulticlassAccuracy`.

.. sgexample:: id3.sg:evaluate_accuracy


----------
References
----------
:wiki:`ID3_algorithm`

:wiki:`Information_gain_in_decision_trees`

.. bibliography:: ../../references.bib
    :filter: docname in docnames

