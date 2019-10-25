===========================
Gaussian Process Classifier
===========================

Application of Gaussian processes in binary and multi-class classification.
`See Gaussian process regression cookbook
<http://shogun.ml/cookbook/latest/examples/gaussian_process/regression.html>`_
and :cite:`Rasmussen2005GPM` for more information on Gaussian processes.

-------
Example
-------

Imagine we have files with training and test data. We create DenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`MulticlassLabels` as

.. sgexample:: classifier.sg:create_features

To fit the input (training) data :math:`\mathbf{X}`, we have to choose appropriate :sgclass:`MeanFunction`
and  :sgclass:`Kernel`. Here we use a basic :sgclass:`ConstMean` and a :sgclass:`GaussianKernel` with chosen width parameter.

.. sgexample:: classifier.sg:create_appropriate_kernel_and_mean_function

We need to specify the inference method to find the posterior distribution of the function values :math:`\mathbf{f}`.
Here we choose to perform Laplace approximation inference method with an instance of :sgclass:`MultiLaplaceInferenceMethod` (See Chapter 18.2 in :cite:`barber2012bayesian` for a detailed introduction)
and pass it the chosen kernel,
the training features, the mean function, the labels and an instance of :sgclass:`SoftMaxLikelihood`,
to specify the distribution of the targets/labels as above.
Finally we create an instance of the :sgclass:`GaussianProcessClassification` classifier.

.. sgexample:: classifier.sg:create_instance

Then we can train the model and evaluate the predictive distribution.
We get predicted :sgclass:`MulticlassLabels`.

.. sgexample:: classifier.sg:train_and_apply

We can extract the probabilities:

.. sgexample:: classifier.sg:extract_the_probabilities

We can evaluate test performance via e.g. :sgclass:`MulticlassAccuracy`.

.. sgexample:: classifier.sg:evaluate_accuracy

----------
References
----------
:wiki:`Gaussian_process`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
