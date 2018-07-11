=======================
Least Square Regression
=======================

A least square regression model can be defined as minimizing the following system:

.. math::
    \frac{1}{2}\left(\sum_{i=1}^N(y_i-{\bf w}\cdot {\bf x}_i)^2\right)

where :math:`N` is the number of training examples, and :math:`x,y` are the training examples and the labels respectively.

This system boils down to solving the linear system:

.. math::
    {\bf w} = \left(\sum_{i=1}^N{\bf x}_i{\bf x}_i^T\right)^{-1}\left(\sum_{i=1}^N y_i{\bf x}_i\right)

The expressed solution is a linear method with bias 0

-------
Example
-------

Suppose we have files with training and test data. We first load the data from the files.

.. sgexample:: lest_square_regression.sg:create_features

We create an instance of :sgclass:`CLeastSquaresRegression` classifier, passing it :math:`\tau`, and training data.

.. sgexample:: lest_square_regression.sg:create_instance

We can then train the regression model and apply it to test data to get the predicted :sgclass:`CRegressionLabels`.

.. sgexample:: lest_square_regression.sg:train_and_apply

Once training is finished, we can extract the weights :math:`{\bf w}`.

.. sgexample:: lest_square_regression.sg:extract_w

Finally, we evaluate the error using :sgclass:`CMeanSquaredError`.

.. sgexample:: lest_square_regression.sg:evaluate_error

----------
References
----------
:wiki:`Ordinary_least_squares`
