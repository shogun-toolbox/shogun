================
Cross Validation
================

Cross validation allows to estimate an algorithm's performance on unseen data.
It is based on training the model on a subset of the training data, and testing on the remaining validation subset of the data.
The process is repeated for several complementary partitions and all validation results are combined, e.g. by averaging.

One type of cross validation is :math:`k-`fold cross validation.
Suppose we have the training data :math:`\mathbf{x}_1,\mathbf{x}_2, ..., \mathbf{x}_{N}` for a positive integer :math:`N`.
:math:`k-`fold cross validation partitions the data into :math:`P_1, P_2, ..., P_n` each with approximately equal size.
Next, we train the model on :math:`P_1, P_2, ...,P_{n-1}` and test it on :math:`P_n` to get performance measure :math:`a_1`, for example :sgclass:`CAccuracyMeasure`, or :sgclass:`MeanSquaredError`.
Finally we train the model again from scratch on :math:`P_1, P_2, ...,P_{n-2}, P_{n}` and test it on :math:`P_{n-1}` to get :math:`a_2`.
This process is repeated until we have :math:`a_1, a_2, ..., a_n`, and then we can compute their mean and standard deviation.

We will perform :math:`k-` fold cross validation with both a classification model (:sgclass:`LibLinear`) and a regression model (:sgclass:`	CLinearRidgeRegression`).

------------------------------
Example for a binary classifier
------------------------------

We use a linear SVM model, see :doc:`../binary/linear_support_vector_machine` for a more complete example of linear SVM usage.

Firstly, we import the training data, training labels, test data, and test labels.

.. sgexample:: cross_validation.sg:create_features

Next,we initialize a splitting strategy :sgclass:`StratifiedCrossValidationSplitting`, which is needed to divide the dataset into :math:`k-`folds for the :math:`k-` fold cross validation.

We also have to decide on an evaluation criterion class (from :sgclass:`CEvaluation`) to evaluate the performance of the trained models.
In this case, we use :sgclass:`CAccuracyMeasure`.
We then instantiate :sgclass:`CrossValidation` and set the number of cross validation's runs.

.. sgexample:: cross_validation.sg:create_cross_validation

Finally, we evaluate the model and get the results (a :sgclass:`CrossValidationResult` instance).
We can also get the mean of all evaluation results and their standard deviation.

.. sgexample:: cross_validation.sg:evaluate_and_get_result


.. sgexample:: cross_validation.sg:get_results

We can then compare it with the accuracy on the test data.

.. sgexample:: cross_validation.sg:get_results_test_data


---------------------
Example for regression
---------------------
We will use the linear ridge regression model. (see :doc:`../regression/linear_ridge_regression` for a more complete example of linear ridge regression usage).

Firstly, we import the training data in the same way done above.
Next, we choose :sgclass:`CrossValidationSplitting` as a splitting strategy (:sgclass:`StratifiedCrossValidationSplitting` only makes sense for classification problems).
Here, we use :sgclass:`MeanSquaredError` as an evaluation criterion.

.. sgexample:: cross_validation.sg:create_cross_validation_REGRESSION

You can evaluate the error of the model and cross validation similar to how it was done in the case of binary .

----------
References
----------

:wiki:`Cross-validation_(statistics)`
