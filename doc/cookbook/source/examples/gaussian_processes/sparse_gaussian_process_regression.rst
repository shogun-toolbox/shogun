==================================
Sparse Gaussian Process Regression
==================================


This cookbook illustrates how to use sparse approximations to Gaussian processes. Sparse approximations for full GP are done to reduce computational scaling. This requires selecting an additional :math:`m` latent variables which could be a  subset of training points. This subset form a pseudo data set with inputs :math:`\mathbf{X}'` and targets :math:`\mathbf{f}'` .

Given the training data, the predictive distribution :math:`y^*` for a new input point :math:`\mathbf{x}^*` will then be:

.. math::
	p(y^*|\mathbf{x}^*, \mathbf{y}, \mathbf{X}, \mathbf{X}')=\int p(\mathbf{y}^*|\mathbf{x}^*, \mathbf{X}',\mathbf{f}')p(\mathbf{f}'| \mathbf{y}, \mathbf{X}, \mathbf{X}')df'

See :cite:`Quinonero-Candela2005` for detailed overview of Sparse approximate Gaussian Processes Regression.

-------
Example
-------

Imagine we have files with training and test data. We create `CDenseFeatures` (here 64 bit floats aka RealFeatures) and :sgclass:`CRegressionLabels` as:

.. sgexample:: sparse_gaussian_process_regression.sg:create_features

To fit the input (training) data :math:`\mathbf{X}`, we have to choose an appropriate :sgclass:`CMeanFunction` and  :sgclass:`CKernel` and instantiate them. Here we use a basic :sgclass:`CZeroMean` and a :sgclass:`CGaussianKernel` with chosen width parameter.

.. sgexample::  sparse_gaussian_process_regression.sg:create_kernel_and_mean_function

We need to specify the inference method to find the posterior distribution of the function values :math:`\mathbf{f}`. Here we choose to perform variational inference for fully independent conditional training (FITC) with an instance of :sgclass:`CFITCInferenceMethod`. We use another feature instance for inducing points and add a simple subset for demonstration. The inference method is then created and we pass it the chosen kernel, the training features, the mean function, the labels, an instance of :sgclass:`CGaussianLikelihood`. We use a subset of the training data for inducing features.

.. sgexample::  sparse_gaussian_process_regression.sg:create_inference

Finally we generate a :sgclass:`CGaussianProcessRegression` class to be trained.

.. sgexample::  sparse_gaussian_process_regression.sg:create_instance

Then we can train the model and evaluate the predictive distribution. We get predicted :sgclass:`CRegressionLabels`.

.. sgexample::  sparse_gaussian_process_regression.sg:train_and_apply

We can compute the predictive variances as

.. sgexample:: sparse_gaussian_process_regression.sg:compute_variance

Finally, we evaluate the :sgclass:`CMeanSquaredError`.

.. sgexample::  sparse_gaussian_process_regression.sg:evaluate_error

----------
References
----------
:wiki:`Gaussian_process`

.. bibliography:: ../../references.bib
	:filter: docname in docnames
