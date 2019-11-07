==============================
Cross Validation on a Pipeline
==============================

In this example, we illustrate how to use cross-validation with :sgclass:`CPipeline`.

For more information on cross-validation, check :doc:`./cross_validation`.

-------
Example
-------
We demonstrate a pipeline consisting of a transformer :sgclass:`CPruneVarSubMean` for normalizing the features, and a machine :sgclass:`LibLinear` for binary classification.

We create :sgclass:`Features` and :sgclass:`Labels` via loading from files

.. sgexample:: cross_validation_pipeline:create_features

We first chain transformers, and then finalize the pipeline with the classifier.

.. sgexample:: cross_validation_pipeline:create_pipeline

Next, we initialize a splitting strategy :sgclass:`StratifiedCrossValidationSplitting` to divide the dataset into :math:`k-` folds for the :math:`k-` fold cross validation.
We also have to decide on an evaluation criterion class (see :sgclass:`CEvaluation`) to evaluate the performance of the trained model.
In this case, we use :sgclass:`CAccuracyMeasure`.
We then instantiate :sgclass:`CrossValidation` and set the number of cross validation's runs.
We next create a cross-validation instance and pass the generated pipeline.

.. sgexample:: cross_validation_pipeline:create_cross_validation

Finally, we evaluate the model and get the results (aka a :sgclass:`CrossValidationResult` instance).

.. sgexample:: cross_validation_pipeline:evaluate_and_get_result

