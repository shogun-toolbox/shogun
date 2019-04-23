==================================
Feedforward Network for Regression
==================================

This page illustrates the usage of feedforward networks for regression. For more details about feedforward networks, see :doc:`feedforward_net_classification`.

-------
Example
-------

Imagine we have files with training and test data. We create DenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`RegressionLabels` as

.. sgexample:: feedforward_net_regression.sg:create_features

We create a :sgclass:`NeuralNetwork` instance by using the above layers and randomly initialize the network parameters by sampling from a gaussian distribution. We set appropriate parameters like regularization coefficient, number of epochs, learning rate, etc. as shown below. More parameters can be found in the documentation of :sgclass:`NeuralNetwork`.

.. sgexample:: feedforward_net_regression.sg:create_instance

We create instances of :sgclass:`NeuralLayers` and add an input layer, hidden layer and output layer which are building blocks of :sgclass:`NeuralNetwork`

.. sgexample:: feedforward_net_regression.sg:add_layers

We train the model and apply it to test data.

.. sgexample:: feedforward_net_regression.sg:train_and_apply

We can extract the parameters of the trained network.

.. sgexample:: feedforward_net_regression.sg:get_params

Finally, we compute :sgclass:`MeanSquaredError`.

.. sgexample:: feedforward_net_regression.sg:evaluate_error

----------
References
----------
:wiki:`Artificial_neural_network`

:doc:`feedforward_net_classification`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
