==========
CHAID tree
==========

CHAID (Chi-squared Automatic Interaction Detector) algorithm is a type of decision tree technique, which relies on the Chi-square test to determine the best next split at each step.

CHAID accepts nominal or ordinal categorical predictors only. If predictors are continuous,
they have to be transformed into ordinal predictors by binning before tree growing, and an ANOVA F-test will be used for nodes split.

-------
Example
-------

Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`CMulticlassLabels` as

.. sgexample:: chaid_tree.sg:create_features

We set the feature types to continuous.
The types can be set to :math:`0` for nominal, :math:`1` for ordinal and :math:`2` for continuous.

.. sgexample:: chaid_tree.sg:set_feature_types

We create an instance of the :sgclass:`CCHAIDTree` classifier by passing it the label and train dataset feature types.
For continuous predictors, the user has to provide the number of bins for continuous to ordinal conversion.

.. sgexample:: chaid_tree.sg:create_instance

Then we train and apply it to test data, which here gives :sgclass:`CMulticlassLabels`.

.. sgexample:: chaid_tree.sg:train_and_apply

We can evaluate test performance via e.g. :sgclass:`CMulticlassAccuracy`.

.. sgexample:: chaid_tree.sg:evaluate_accuracy

----------
References
----------

:wiki:`CHAID`

:wiki:`Chi-squared_test`

:wiki:`Analysis_of_variance`