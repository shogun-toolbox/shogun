=======================
Least Angle Regression
=======================

Least Angle Regression (LARS) is an algorithm used to fit a linear regression model. LARS is simliar to forward stagewise regression but less greedy. Instead of including variables at each step, the estimated parameters are increased in a direction equiangular to each one's correlations with the residual. LARS can be used to solve LASSO, which is L1-regularized least square regression.

.. math::
	\min \|X^T\beta - y\|^2 + \lambda\|\beta\|_{1}]

	\|\beta\|_1 = \sum_i|\beta_i|

where :math:`X` is the feature matrix with explanatory features and :math:`y` is the dependent variable to be predicted. 
Pre-processing of :math:`X` and :math:`y` are needed to ensure the correctness of this algorithm:
:math:`X` needs to be normalized: each feature should have zero-mean and unit-norm, 
:math:`y` needs to be centered: its mean should be zero.


-------
Example
-------

Imagine we have files with training and test data. We create `CDenseFeatures` (here 64 bit floats aka RealFeatures) and :sgclass:`CRegressionLabels` as

.. sgexample:: least_angle_regression.sg:create_features

To normalize and center the features, we create an instance of preprocessors :sgclass:`CPruneVarSubMean` and :sgclass:`CNormOne` and apply it on the feature matrices.

.. sgexample:: least_angle_regression:preprocess_features

We create an instance of :sgclass:`CLeastAngleRegression` by selecting to disable the LASSO solution, setting the penalty :math:`\lambda` for l1 norm and setting training data and labels.

.. sgexample:: least_angle_regression:create_instance

Then we train the regression model and apply it to test data to get the predicted :sgclass:`CRegressionLabels` .

.. sgexample:: linear_ridge_regression.sg:train_and_apply

After training, we can extract :math:`{\bf w}`.

.. sgexample:: linear_ridge_regression.sg:extract_w

Finally, we can evaluate the :sgclass:`CMeanSquaredError`.

.. sgexample:: linear_ridge_regression.sg:evaluate_error

----------
References
----------
:wiki:`Least-angle_regression`

:wiki:`Stepwise_regression`
