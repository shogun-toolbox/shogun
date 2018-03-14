============================================
Cross Validation
============================================

Cross Validation is a type of model validation technique for figuring out how the
results from a statistical analysis will generalize to an independent data set. It
essentially involves training the model on a subset of the data, and testing on the
remainder, then repeating this process for several complementary partitions and then
combining all the validation results generated (For instance, an average) to estimate
a performance for the final predictive model.

One type of cross validation is known as :math:`k-` fold cross validation.

For instance, Suppose you have the training data :math:`\mathbf{x}_1,\mathbf{x}_2, ..., \mathbf{x}_{10n}`
for a positive integer :math:`n` , then :math:`10-` fold cross validation might for instance partition
the data into :math:`P_1, P_2, ..., P_n` each with :math:`10` data points.

Next, the model would be trained on :math:`P_1, P_2, ...,P_{n-1}` and is tested on :math:`P_n` to get accuracy
:math:`a_1` .

Then the model would train again from scratch on :math:`P_1, P_2, ...,P_{n-2}, P_{n}` and is tested on :math:`P_{n-1}`
to get :math:`a_2`.

This process is repeated until we have :math:`a_1, a_2, ..., a_n`, at which we take their average and calculate
their standard deviation.

We will perform :math:`k-` fold cross-validation with both a classification model (Linear SVM Model) and a
regression model (Linear Ridge Regression).

-------
Example of :math:`k-` Fold Cross-Validation on a Binary Classifier
-------
We will use Linear SVM Model. (see :doc:`../binary/linear_support_vector_machine` for a more complete example of Linear SVM usage).

Firstly, we import the training data, training labels, test data, and test labels.

.. sgexample:: cross_validation.sg:create_features

Next,we initialize a splitting strategy :sgclass:`CStratifiedCrossValidationSplitting`, which is needed
to divide the dataset into :math:`k-` folds for the :math:`k-` fold cross validation (In this case, we make :math:`k=2` ).

We also have to decide on an evaluation criterion class (From :sgclass:`CEvaluation`) to evaluate the performance of the trained models. For this binary classifier,
we use :sgclass:`CAccuracyMeasure` to evaluate the accuracy of the model.

Finally, we instantiate a :sgclass:`CCrossValidation` instance. We also set the number of cross validation's runs.

.. sgexample:: cross_validation.sg:create_cross_validation

Finally, we evaluate the model and get the results (a :sgclass:`CCrossValidationResult` instance).

.. sgexample:: cross_validation.sg:evaluate_and_get_result

We get the mean of all the evaluation results and their standard deviation stddev.

.. sgexample:: cross_validation.sg:get_results

We can then compare it with the accuracy on the test data.

.. sgexample:: cross_validation.sg:get_results_test_data



-------
Example of :math:`k-` Fold Cross-Validation on Regression
-------
We will use the Linear Ridge Regression model. (see :doc:`../regression/linear_ridge_regression` for a more
complete example of Linear Ridge Regression usage).

Firstly, we import the training data, training labels, test data, and test labels.

.. sgexample:: cross_validation.sg:create_features_REGRESSION

Next,we initialize a splitting strategy :sgclass:`CCrossValidationSplitting` (Do not use :sgclass:`CStratifiedCrossValidationSplitting`
with Regression), which is needed to divide the dataset into :math:`k-` folds for the :math:`k-` fold cross validation (In this case,
we make :math:`k=2` ).

We also have to decide on an evaluation criterion class (From :sgclass:`CEvaluation`) to evaluate the performance of the trained models.
For this regression classifier, we use :sgclass:`CMeanSquaredError` to evaluate the accuracy of the model.

Finally, we instantiate a :sgclass:`CCrossValidation` instance. We also set the number of cross validation's runs.

.. sgexample:: cross_validation.sg:create_cross_validation_REGRESSION

Finally, we evaluate the model and get the results (a :sgclass:`CCrossValidationResult` instance).

.. sgexample:: cross_validation.sg:evaluate_and_get_result_REGRESSION

We get the mean of all the mean square errors and their standard deviation stddev.

.. sgexample:: cross_validation.sg:get_results_REGRESSION

We can then compare it with the mean square error on the test data.

.. sgexample:: cross_validation.sg:evaluate_error_REGRESSION


----------
References
----------

:wiki:`Cross-validation_(statistics)`
