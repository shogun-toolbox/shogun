=======================
Kernel Ridge Regression
=======================

Kernel ridge regression is a non-parametric form of ridge regression. The aim is to learn a function in the space induced by the respective kernel :math:`k` by minimizing a squared loss with a squared norm regularization term.

The solution can be written in closed form as:

.. math::
    \alpha = \left({\bf K}+\tau{\bf I}\right)^{-1}{\bf y}

where :math:`{\bf K}` is the kernel matrix and :math:`\alpha` is the vector of weights in the space induced by the kernel.
The learned function can then be evaluated as :math:`f(x)=\sum_{i=1}^N\alpha_ik(x,x_i)`.

See Chapter 17 in :cite:`barber2012bayesian` for a detailed introduction.

-------
Example
-------

Imagine we have files with training and test data. We create `CDenseFeatures` (here 64 bit floats aka RealFeatures) and :sgclass:`CRegressionLabels` as

.. sgexample:: kernel_ridge_regression.sg:create_features

Choose an appropriate :sgclass:`CKernel` and instantiate it. Here we use a :sgclass:`CGaussianKernel`.

.. sgexample:: kernel_ridge_regression.sg:create_appropriate_kernel

We create an instance of :sgclass:`CKernelRidgeRegression` classifier by passing it :math:`\tau`, the kernel and labels.

.. sgexample:: kernel_ridge_regression.sg:create_instance

Then we train the regression model and apply it to test data to get the predicted :sgclass:`CRegressionLabels`.

.. sgexample:: kernel_ridge_regression.sg:train_and_apply

After training, we can extract :math:`\alpha`.

.. sgexample:: kernel_ridge_regression.sg:extract_alpha

Finally, we can evaluate the :sgclass:`CMeanSquaredError`.

.. sgexample:: kernel_ridge_regression.sg:evaluate_error

----------
References
----------
:wiki:`Kernel_method`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
