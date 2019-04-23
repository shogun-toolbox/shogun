=======================
Linear Ridge Regression
=======================

A linear ridge regression model can be defined as :math:`y_i = \bf{w}^\top\bf{x_i}` where :math:`y_i` is the predicted value, :math:`\bf{x_i}` is a feature vector, :math:`\bf{w}` is the weight vector.
We aim to find the linear function that best explains the data, i.e. minimizes the squared loss plus a :math:`L_2` regularization term. One can show the solution can be written as:

.. math::
    {\bf w}=\left(\tau I_{D}+XX^{\top}\right)^{-1}X^{\top}y

where :math:`X=\left[{\bf x}_{1},\dots{\bf x}_{N}\right]\in\mathbb{R}^{D\times N}` is the training data matrix, containing :math:`N` training samples of dimension :math:`D`, :math:`y=[y_{1},\dots,y_{N}]^{\top}\in\mathbb{R}^{N}` are the labels, and :math:`\tau>0` scales the regularization term.

Alternatively if :math:`D>N`, the solution can be written as
.. math::
    {\bf w}=X\left(\tau I_{N}+X^{\top}X\right)^{-1}y

In practice, an additional bias :math:`b=\frac{1}{N}\sum_{i=1}^{N}y_{i}-{\bf w}\cdot\bar{\mathbf{x}}` for
:math:`\bar{\mathbf{x}}=\frac{1}{N}\sum_{i=1}^{N}{\bf x}_{i}` can also be included, which effectively centers the :math:`X` before
computing the solution.

For the special case when :math:`\tau = 0`, a wrapper class :sgclass:`LeastSquaresRegression` is available.

-------
Example
-------

Imagine we have files with training and test data. We create `DenseFeatures` (here 64 bit floats aka RealFeatures) and :sgclass:`RegressionLabels` as

.. sgexample:: linear_ridge_regression.sg:create_features

We create an instance of :sgclass:`CLinearRidgeRegression` classifier, passing it :math:`\tau`, training data and labels.

.. sgexample:: linear_ridge_regression.sg:create_instance

Then we train the regression model and apply it to test data to get the predicted :sgclass:`RegressionLabels`.

.. sgexample:: linear_ridge_regression.sg:train_and_apply

After training, we can extract :math:`{\bf w}` and the bias.

.. sgexample:: linear_ridge_regression.sg:extract_w

We could also have trained without bias and set it manually.

.. sgexample:: linear_ridge_regression.sg:manual_bias

Finally, we can evaluate the :sgclass:`MeanSquaredError`.

.. sgexample:: linear_ridge_regression.sg:evaluate_error

----------
References
----------
:wiki:`Tikhonov_regularization`

:wiki:`Ordinary_least_squares`
