=============================
Perceptron
=============================

The perceptron is an algorithm for binary classification. It maps its input :math:`x` to output value:math:`y` using the conditions:

.. math::

	y = \begin{cases}
	  0, & \text{if } {\bf w}.x + b > 0, \\
	  1, & \text{otherwise}.
	\end{cases}

The algorithm allows for online learning by processing the elements in training set one at a time.

See Chapter 17 in :cite:`barber2012bayesian` for a detailed introduction.

-------
Example
-------

Imagine we have files with training and test data. We create DenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`BinaryLabels` as

.. sgexample:: perceptron.sg:create_features

We need to initialize parameters for maximum number of iterations (since perceptron is not guarenteed to converge) and learning rate.

.. sgexample:: perceptron.sg:set_parameters

We create an instance of the :sgclass:`Perceptron` classifier by passing it training features and labels.

.. sgexample:: perceptron.sg:create_instance

Then we train and apply it to test data, which here gives :sgclass:`BinaryLabels`.

.. sgexample:: perceptron.sg:train_and_apply

We can extract :math:`{\bf w}` and :math:`b`.

.. sgexample:: perceptron.sg:extract_weights_bias

We can evaluate test performance via e.g. :sgclass:`CAccuracyMeasure`.

.. sgexample:: perceptron.sg:evaluate_accuracy

----------
References
----------
:wiki:`Perceptron`

.. bibliography:: ../../references.bib
    :filter: docname in docnames

