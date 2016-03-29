=============================
Linear Support Vector Machine
=============================

Linear Support Vector Machine is a binary classifier which finds a hyper-plane such that the margins between the two classes are maximized. The loss function that needs to be minimized is:

.. math::

    \min_{\bf w} \frac{1}{2}{\bf w}^\top{\bf w} + C\sum_{i=1}^{N}\xi({\bf w};{\bf x_i}, y_i)

where :math:`{\bf w}` is vector of weights, :math:`{\bf x_i}` is feature vector, :math:`y_i` is the corresponding label, :math:`C>0` is a penalty parameter, :math:`N` is the number of training samples and :math:`\xi` is hinge loss function.

The solution takes the following form:

.. math::

    \mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i

:math:`\alpha_i` are sparse in the above solution.

See :cite:`fan2008liblinear` and Chapter 6 in :cite:`cristianini2000introduction` for a detailed introduction.

-------
Example
-------

Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`CBinaryLabels` as

.. sgexample:: linear_svm.sg:create_features

In order to run :sgclass:`CLibLinear`, we need to initialize some parameters like :math:`C` and epsilon which is the residual convergence parameter of the solver.

.. sgexample:: linear_svm.sg:set_parameters

We create an instance of the :sgclass:`CLibLinear` classifier by passing it regularization coefficient, features and labels. We here set the solver type to L2 regularized classification. There are many other solver types in :sgclass:`CLibLinear` to choose from.

.. sgexample:: linear_svm.sg:create_instance

Then we train and apply it to test data, which here gives :sgclass:`CBinaryLabels`.

.. sgexample:: linear_svm.sg:train_and_apply

We can extract :math:`{\bf w}` and :math:`b`.

.. sgexample:: linear_svm.sg:extract_weights_bias

We can evaluate test performance via e.g. :sgclass:`CAccuracyMeasure`.

.. sgexample:: linear_svm.sg:evaluate_accuracy

----------
References
----------
:wiki:`Support_vector_machine`

:wiki:`Lagrange_multiplier`

`LibLinear website <http://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_

.. bibliography:: ../../references.bib
    :filter: docname in docnames
