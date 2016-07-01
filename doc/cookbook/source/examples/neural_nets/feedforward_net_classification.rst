======================================
Feedforward Network for Classification
======================================

Feedforward network or multi-layer perceptron defines a mapping :math:`y = f(\mathbf{x};\mathbf{\theta})` from an input :math:`\mathbf{x}` to a category :math:`y` and learns the value of parameters :math:`\mathbf{\theta}` by iterative training that results in the best function approximation. The network is a directed acyclic graph composed of an input layer, an output layer and a few hidden layers.

For example,

.. math::

    f(\mathbf{x}) = f^{(3)}(f^{(2)}(f^{(1)}(\mathbf{x})))

where :math:`\mathbf{x}` is the input layer, :math:`f^{(1)}` and :math:`f^{(2)}` are hidden layers and :math:`f^{(3)}` is the output layer.

See chapter 6 in :cite:`Goodfellow-et-al-2016-Book` for a detailed introduction.

-------
Example
-------

Imagine we have files with training and test data. We create CDenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`CBinaryLabels` as

.. sgexample:: feedforward_net_classification.sg:create_features

We create instances of :sgclass:`CNeuralInputLayer`, :sgclass:`CNeuralLinearLayer` and :sgclass:`NeuralSoftmaxLayer` which are building blocks of :sgclass:`CNeuralNetwork`

.. sgexample:: feedforward_net_classification.sg:add_layers

We create a :sgclass:`CNeuralNetwork` instance by using the above layers and randomly initialize the network parameters by sampling from a gaussian distribution.

.. sgexample:: feedforward_net_classification.sg:create_instance

Before training, we need to set appropriate parameters like regularization coefficient, dropout probabilities, learning rate, etc. as shown below. More parameters can be found in the documentation of :sgclass:`CNeuralNetwork`.

.. sgexample:: feedforward_net_classification.sg:set_parameters

We train the model and apply it to some test data.

.. sgexample:: feedforward_net_classification.sg:train_and_apply

We can extract the parameters of the trained network.

.. sgexample:: feedforward_net_classification.sg:get_params

Finally, we compute accuracy.

.. sgexample:: feedforward_net_classification.sg:evaluate_accuracy

----------
References
----------
:wiki:`Artificial_neural_network`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
