===============================
Multi-class Logistic Regression
===============================

Multinomial logistic regression assigns the sample :math:`\mathbf{x}_i` to class :math:`c`
based on the probability for sample :math:`\mathbf{x}_i` to be in class :math:`c`:

.. math::

    P(Y_i = c | \mathbf{x}_i) = \frac{\exp(\mathbf{\theta}^\top_c\mathbf{x}_i)}{1+ \sum_{k=1}^{K}\exp(\mathbf{\theta}^\top_k\mathbf{x}_i)}

in which :math:`K` is the number of classes.

The loss function that needs to be minimized is:

.. math::

    {\min_{\mathbf{\theta}}}\sum_{k=1}^{K}\sum_{i=1}^{m}w_{ik}\log(1+\exp(-y_{ik}(\mathbf{x}_k^\top\mathbf{a}_{ik} + c_k))) + \lambda\left \| \mathbf{x} \right \|_{l_1/l_q}

where :math:`\mathbf{a}_{ik}` denotes the :math:`i`-th sample for the :math:`k`-th class, :math:`w_{ik}` is the weight for :math:`\mathbf{a}_{ik}^\top`,
:math:`y_{ik}` is the response of :math:`\mathbf{a}_{ik}`, and :math:`c_k` is the intercept (scalar) for the :math:`k`-th class. 
:math:`\lambda` is the :math:`l_1/l_q`-norm regularization parameter.

-------
Example
-------

Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`CMulticlassLabels` as

.. sgexample:: multiclass_logisticregression.sg:create_features

We create an instance of the :sgclass:`CMulticlassLogisticRegression` classifier by passing it the dataset, lables, and specifying the regularization constant :math:`\lambda` for each machine

.. sgexample:: multiclass_logisticregression.sg:create_instance

Then we train and apply it to test data, which here gives :sgclass:`CMulticlassLabels`.

.. sgexample:: multiclass_logisticregression.sg:train_and_apply

We can evaluate test performance via e.g. :sgclass:`CMulticlassAccuracy`.

.. sgexample:: multiclass_logisticregression.sg:evaluate_accuracy

----------
References
----------

:wiki:`Multinomial_logistic_regression`

:wiki:`Multiclass_classification`
