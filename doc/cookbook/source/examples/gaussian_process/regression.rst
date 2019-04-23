===========================
Gaussian Process Regression
===========================

A Gaussian Process is the extension of the Gaussian distribution to infinite dimensions. It is specified by a mean function :math:`m(\mathbf{x})` and a covariance kernel :math:`k(\mathbf{x},\mathbf{x}')` (where :math:`\mathbf{x}\in\mathcal{X}` for some input domain :math:`\mathcal{X}`). It defines a distribution over real valued functions :math:`f(\cdot)`. Any finite set of such function values :math:`\mathbf{f}` is jointly Gaussian with mean specified by the mean function evaluated at the inputs :math:`\mathbf{x}_i` and covariance matrix :math:`\mathbf{C}` with entries :math:`\mathbf{C}_{ij}=k(\mathbf{x}_i,\mathbf{x}_j)`.

In our regression model, we take our target variable :math:`\mathbf{y}` to be our (latent) function values :math:`\mathbf{f}` plus Gaussian noise:

.. math::
	p(y_i|f_i)=\mathcal{N}(f_i,\sigma^2)

Given the training data, the prediction :math:`y^*` for a new input point :math:`\mathbf{x}^*` is distributed as:

.. math::
	p(y^*|\mathbf{x}^*, \mathbf{y}, \mathbf{X})=\int p(\mathbf{y}^*|f^*)p(f^*|\mathbf{x}^*, \mathbf{f})p(\mathbf{f}|\mathbf{y}, \mathbf{X})d\mathbf{f}df^*

This is known as the "predictive" distribution.

See :cite:`Rasmussen2005GPM` for a comprehensive treatment of Gaussian Processes (see Chapter 2 for regression).

-------
Example
-------

Imagine we have files with training and test data. We create `DenseFeatures` (here 64 bit floats aka RealFeatures) and :sgclass:`RegressionLabels` as:

.. sgexample:: regression.sg:create_features

To fit the input (training) data :math:`\mathbf{X}`, we have to choose an appropriate :sgclass:`MeanFunction` and  :sgclass:`Kernel` and instantiate them. Here we use a basic :sgclass:`ZeroMean` and a :sgclass:`GaussianKernel` with chosen width parameter.

.. sgexample:: regression.sg:create_appropriate_kernel_and_mean_function

We need to specify the inference method to find the posterior distribution of the function values :math:`\mathbf{f}`. Here we choose to perform exact inference with an instance of :sgclass:`ExactInferenceMethod` and pass it the chosen kernel, the training features, the mean function, the labels and an instance of :sgclass:`GaussianLikelihood`, to specify the distribution of the targets/labels as above. Finally we generate a GaussianProcessRegression class to be trained.

.. sgexample:: regression.sg:create_instance

Then we can train the model and evaluate the predictive distribution. We get predicted :sgclass:`RegressionLabels`.

.. sgexample:: regression.sg:train_and_apply

We can compute the predictive variances as

.. sgexample:: regression.sg:compute_variance

The prediction above is based on arbitrarily set hyperparameters :math:`\boldsymbol{\theta}`: kernel width :math:`\tau`, kernel scaling :math:`\gamma` and observation noise :math:`\sigma^2`. We can also learn these parameters by optimizing the marginal likelihood :math:`p(\mathbf{y}|\mathbf{X}, \boldsymbol{\theta})` w.r.t. :math:`\boldsymbol{\theta}`.
To do this, we define a :sgclass:`CGradientModelSelection`, passing to it a :sgclass:`CGradientEvaluation` with its own :sgclass:`CGradientCriterion`, specifying the gradient scheme and direction. Then we can follow the gradient and apply the chosen :math:`\boldsymbol{\theta}` back to the GaussianProcessRegression instance.

.. sgexample:: regression.sg:optimize_marginal_likelihood

Finally, we evaluate the :sgclass:`MeanSquaredError` and the (negative log) marginal likelihood for the optimized hyperparameters.

.. sgexample:: regression.sg:evaluate_error_and_marginal_likelihood

----------
References
----------
:wiki:`Gaussian_process`

.. bibliography:: ../../references.bib
	:filter: docname in docnames
