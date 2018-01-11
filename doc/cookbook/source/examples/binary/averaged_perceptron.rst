===================
Averaged Perceptron
===================

The averaged Perceptron is an online binary classifier. It is an extension
of the standard Perceptron algorithm; it uses the `averaged` weight and
bias.
Given a vector :math:`\mathbf{x}`, the predicted class is given by:

.. math::

    \theta\left(\mathbf{w} \cdot \mathbf{x}+b\right)

Here, :math:`\mathbf{w}` is the average weight vector,
:math:`b` is the average bias and :math:`\theta` is a step function:

.. math::

    \theta(x) =
    \begin{cases}
    1 & x > 0 \\
    0 & x = 0 \\
    -1 & x < 0
    \end{cases}

See chapter 17 in :cite:`barber2012bayesian` for a brief explanation of the Perceptron.

-------
Example
-------

Given a linearly separable dataset, we create some CDenseFeatures
(RealFeatures, here 64 bit float values) and some :sgclass:`CBinaryLabels` to set up the training and validation sets.

.. sgexample:: averaged_perceptron.sg:create_features

We create the :sgclass:`CAveragedPerceptron` instance by passing it the traning features and labels.
We also set its learn rate and its maximum iterations.

.. sgexample:: averaged_perceptron.sg:set_parameters

Then we train the :sgclass:`CAveragedPerceptron` and we apply it to the test data, which gives the predicted :sgclass:`CBinaryLabels`.

.. sgexample:: averaged_perceptron.sg:train_and_apply

We can also extract the average weights :math:`\mathbf{w}` and the bias :math:`b`.

.. sgexample:: averaged_perceptron.sg:extract_weights

Finally, we can evaluate the performance, e.g. using :sgclass:`CAccuracyMeasure`.

.. sgexample:: averaged_perceptron.sg:evaluate_accuracy

----------
References
----------
:wiki:`Perceptron`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
