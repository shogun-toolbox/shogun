=======================
Linear Ridge Regression
=======================

A linear ridge regression model can be defined as :math:`y_i = \bf{w}^\top\bf{x_i}` where :math:`y_i` is the predicted value, :math:`\bf{x_i}` is a feature vector and :math:`\bf{w}` is the weight vector. We aim to find the linear function that best explains the data, i.e. minimizes the squared loss plus a :math:`L_2` regularization term. One can show the solution can be written as:

.. math::
    {\bf w} = \left(\tau {\bf I}+ \sum_{i=1}^N{\bf x}_i{\bf x}_i^\top\right)^{-1}\left(\sum_{i=1}^N y_i{\bf x}_i\right)

where :math:`N` is the number of training samples and :math:`\tau>0` scales the regularization term.

A bias term, which is the squared empirical error, can also be calculated.

For the special case when :math:`\tau = 0`, a wrapper class :sgclass:`CLeastSquaresRegression` is available.

-------
Example
-------

Imagine we have files with training and test data. We create `CDenseFeatures` (here 64 bit floats aka RealFeatures) and :sgclass:`CRegressionLabels` as

.. sgexample:: linear_ridge_regression.sg:create_features

We create an instance of :sgclass:`CLinearRidgeRegression` classifier, passing it :math:`\tau`, training data and labels.

.. sgexample:: linear_ridge_regression.sg:create_instance

Then we train the regression model and apply it to test data to get the predicted :sgclass:`CRegressionLabels` and bias.

.. sgexample:: linear_ridge_regression.sg:train_and_apply

Optionally the bias can be disabled to avoid redundant computation.

.. sgexample:: linear_ridge_regression.sg:disable_bias

Imagine, we know the bias term. We can set it as

.. sgexample:: linear_ridge_regression.sg:set_bias_manually

After training, we can extract :math:`{\bf w}`.

.. sgexample:: linear_ridge_regression.sg:extract_w

Finally, we can evaluate the :sgclass:`CMeanSquaredError`.

.. sgexample:: linear_ridge_regression.sg:evaluate_error

----------
References
----------
:wiki:`Tikhonov_regularization`

:wiki:`Ordinary_least_squares`
