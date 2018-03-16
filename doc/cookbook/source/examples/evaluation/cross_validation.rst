============================================
Cross Validation
============================================

Cross validation is a type of model validation technique to estimate an algorithm's performance on unseen data.

It essentially involves training the model on a subset of the data, and testing on the
remaining data.

The process is repeated for several complementary partitions and
combines all the validation results generated (For instance, an average).

One type of cross validation is known as :math:`k-` fold cross validation.

Suppose we have the training data :math:`\mathbf{x}_1,\mathbf{x}_2, ..., \mathbf{x}_{N}`
for a positive integer :math:`n` , then :math:`k-` fold cross validation partitions
the data into :math:`P_1, P_2, ..., P_n` each with equal size (Where :math:`n=\frac{N}{k}`).

Next, the model would be trained on :math:`P_1, P_2, ...,P_{n-1}` and is tested on :math:`P_n` to get performance measure
:math:`a_1` .

Then the model would train again from scratch on :math:`P_1, P_2, ...,P_{n-2}, P_{n}` and is tested on :math:`P_{n-1}`
to get :math:`a_2`.

This process is repeated until we have :math:`a_1, a_2, ..., a_n`, at which we can compute their mean and standard deviation.

We perform :math:`k-` fold cross-validation with both a classification model (linear SVM) and a
regression model (linear ridge regression).

-------
Example of a binary Classifier
-------

We use a linear SVM model. (see :doc:`../binary/linear_support_vector_machine` for a more complete example of linear SVM usage).

Firstly, we import the training data, training labels, test data, and test labels.

.. sgexample:: cross_validation.sg:create_features

Next,we initialize a splitting strategy :sgclass:`CStratifiedCrossValidationSplitting`, which is needed
to divide the dataset into :math:`k-` folds for the :math:`k-` fold cross validation.

We also have to decide on an evaluation criterion class (from :sgclass:`CEvaluation`) to evaluate the performance of the trained models.

In this case, we use :sgclass:`CAccuracyMeasure`

We then instantiate a :sgclass:`CCrossValidation` instance and set the number of cross validation's runs.

.. sgexample:: cross_validation.sg:create_cross_validation

Finally, we evaluate the model and get the results (a :sgclass:`CCrossValidationResult` instance).

.. sgexample:: cross_validation.sg:evaluate_and_get_result

We can also get the mean of all evaluation results and their standard deviation.

.. sgexample:: cross_validation.sg:get_results

We can then compare it with the accuracy on the test data.

.. sgexample:: cross_validation.sg:get_results_test_data


-------
Example of regression
-------
We will use the linear ridge regression model. (see :doc:`../regression/linear_ridge_regression` for a more
complete example of linear ridge regression usage).

Firstly, we import the training data, training labels, test data, and test labels.

.. sgexample:: cross_validation.sg:create_features_REGRESSION

Next,we initialize a splitting strategy :sgclass:`CCrossValidationSplitting` (Do not use :sgclass:`CStratifiedCrossValidationSplitting`
with regression), which is needed to divide the dataset into :math:`k-` folds for the :math:`k-` fold cross validation

We also have to decide on an evaluation criterion class (from :sgclass:`CEvaluation`) to evaluate the performance of the trained models.

Here, we use :sgclass:`CMeanSquaredError`.

We then instantiate a :sgclass:`CCrossValidation` instance and set the number of cross validation's runs.

.. sgexample:: cross_validation.sg:create_cross_validation_REGRESSION

Finally, we evaluate the model and get the results (a :sgclass:`CCrossValidationResult` instance).

.. sgexample:: cross_validation.sg:evaluate_and_get_result_REGRESSION

We can also get the mean of all mean square errors and their standard deviation.

.. sgexample:: cross_validation.sg:get_results_REGRESSION

Then we can compare it with the mean square error on the test data.

.. sgexample:: cross_validation.sg:evaluate_error_REGRESSION


----------
References
----------

:wiki:`Cross-validation_(statistics)`
