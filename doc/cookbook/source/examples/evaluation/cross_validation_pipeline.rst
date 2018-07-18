============================
Cross Validation on Pipeline
============================

In this example, we illustrate how to use cross-validation with :sgclass:`CPipeline`.

For more information on cross-validation, check :doc:`./cross_validation`.

-------
Example
-------
We'll use as example a binary classification problem solvable by a pipeline consisted of a transformer :sgclass:`CPruneVarSubMean` and a machine :sgclass:`CLibLinear`.

Imagine we have files with training data. We create :sgclass:`CDenseFeatures` (here 64 bit floats aka RealFeatures) as

.. sgexample:: cross_validation_pipeline:create_features


We use :sgclass:`CPruneVarSubMean` to normalize the features and then use :sgclass:`CLibLinear` for classification.
The transformer and the machine are chained as a :sgclass:`CPipeline`.

.. sgexample:: cross_validation_pipeline:create_pipeline

Next, we initialize a splitting strategy :sgclass:`CStratifiedCrossValidationSplitting` to divide the dataset into :math:`k-` folds for the :math:`k-` fold cross validation.
We also have to decide on an evaluation criterion class (from :sgclass:`CEvaluation`) to evaluate the performance of the trained models.
In this case, we use :sgclass:`CAccuracyMeasure`.
We then instantiate :sgclass:`CCrossValidation` and set the number of cross validation's runs.

.. sgexample:: cross_validation_pipeline:create_cross_validation

Finally, we evaluate the model and get the results (aka a :sgclass:`CCrossValidationResult` instance).

.. sgexample:: cross_validation_pipeline:evaluate_and_get_result

