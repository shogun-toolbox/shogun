=======================
Linear Ridge Regression
=======================

A linear ridge regression model can be defined as :math:`y_i = \bf{w}^\top\bf{x_i} + b` where :math:`y_i` is the predicted value, :math:`\bf{x_i}` is a feature vector, :math:`\bf{w}` is the weight vector, and :math:`b` is a bias term.
We aim to find the linear function that best explains the data, i.e. minimizes the squared loss plus a :math:`L_2` regularization term. One can show the solution can be written as:

.. math::
    {\bf w}=\left(\tau I_{D}+XX^{\top}\right)^{-1}X^{\top}y

where :math:`X=\left[{\bf x}_{1},\dots{\bf x}_{N}\right]\in\mathbb{R}^{D\times N}` is the training data matrix, containing :math:`N` training samples of dimension :math:`D`, :math:`y=[y_{1},\dots,y_{N}]^{\top}\in\mathbb{R}^{N}` are the labels, and :math:`\tau>0` scales the regularization term.

The bias term is computed as :math:`b=\frac{1}{N}\sum_{i=1}^{N}y_{i}-{\bf w}\cdot\bar{\mathbf{x}}`, where :math:`\bar{\mathbf{x}}=\frac{1}{N}\sum_{i=1}^{N}{\bf x}_{i}`.

For the special case when :math:`\tau = 0`, a wrapper class :sgclass:`CLeastSquaresRegression` is available.

-------
Example
-------

Imagine we have files with training and test data. We create `CDenseFeatures` (here 64 bit floats aka RealFeatures) and :sgclass:`CRegressionLabels` as

.. sgexample:: linear_ridge_regression.sg:create_features

We create an instance of :sgclass:`CLinearRidgeRegression` classifier, passing it :math:`\tau`, training data and labels.

.. sgexample:: linear_ridge_regression.sg:create_instance

Then we train the regression model and apply it to test data to get the predicted :sgclass:`CRegressionLabels`.

.. sgexample:: linear_ridge_regression.sg:train_and_apply

After training, we can extract :math:`{\bf w}` and the bias.

.. sgexample:: linear_ridge_regression.sg:extract_w

Finally, we can evaluate the :sgclass:`CMeanSquaredError`.

.. sgexample:: linear_ridge_regression.sg:evaluate_error

----------
References
----------
:wiki:`Tikhonov_regularization`

:wiki:`Ordinary_least_squares`
