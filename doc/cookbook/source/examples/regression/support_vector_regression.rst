=========================
Support Vector Regression
=========================

Support vector regression is a regression model inspired from support vector machines. The solution can be written as:

.. math::
    f({\bf x})=\sum_{i=1}^{N} \alpha_i k({\bf x}, {\bf x}_i)+b

where :math:`{\bf x}` is the new data point, :math:`{\bf x}_i` is a training sample, :math:`N` denotes number of training samples, :math:`k` is a kernel function, :math:`\alpha` and :math:`b` are determined in training.

See :cite:`scholkopf2002learning` for a more detailed introduction. :sgclass:`LibSVR` performs support vector regression using LibSVM :cite:`chang2011libsvm`.

-------
Example
-------

Imagine we have files with training and test data. We create `CDenseFeatures` (here 64 bit floats aka RealFeatures) and :sgclass:`CRegressionLabels` as

.. sgexample:: support_vector_regression.sg:create_features

Choose an appropriate :sgclass:`CKernel` and instantiate it. Here we use a :sgclass:`CGaussianKernel`.

.. sgexample:: support_vector_regression.sg:create_appropriate_kernel

We create an instance of :sgclass:`CLibSVR` classifier by passing it the kernel, labels, solver type and some more parameters. More solver types are available in :sgclass:`CLibSVR`. See :cite:`chang2002training` for more details.

.. sgexample:: support_vector_regression.sg:create_instance

Then we train the regression model and apply it to test data to get the predicted :sgclass:`CRegressionLabels`.

.. sgexample:: support_vector_regression.sg:train_and_apply

After training, we can extract :math:`\alpha`.

.. sgexample:: support_vector_regression.sg:extract_alpha

Finally, we can evaluate the :sgclass:`CMeanSquaredError`.

.. sgexample:: support_vector_regression.sg:evaluate_error

----------
References
----------
:wiki:`Support_vector_machine`

.. bibliography:: ../../references.bib
    :filter: docname in docnames