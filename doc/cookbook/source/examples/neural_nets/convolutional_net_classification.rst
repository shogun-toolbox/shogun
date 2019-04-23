===============================================
Convolutional Neural Network for Classification
===============================================

Convolutional neural network  is a class of deep, feed-forward artificial neural networks, most commonly applied to analyzing visual imagery. 
The network is a directed acyclic graph composed of an input layer, a few hidden layers can be convolutional layers, pooling layers, fully connected layers or normalization layers and a softmax output layer.
To compute the pre-nonlinearity input to some unit :math:`x_{ij}^{l}` in any layer, we can sum up the contributions from the previous layer cells:

.. math::

    x_{ij}^{l}=\sum_{a=0}^{m-1}\sum_{b=0}^{m-1}w_{ab}y_{(i+1)(j+1)}^{l-1}

where :math:`x_{ij}^{l}` is the input to a neuron in layer :math:`l`, :math:`y_{ij}^{l}` is output of the neuron and :math:`w_{ij}` is it's weight.
See chapter 9 in :cite:`Goodfellow-et-al-2016-Book` for a detailed introduction.

-------
Example
-------

Imagine we have files with training and test data. We create DenseFeatures (here 64 bit floats aka RealFeatures) and :sgclass:`MulticlassLabels` as

.. sgexample:: convolutional_net_classification.sg:create_features

We create a :sgclass:`NeuralNetwork` instance and randomly initialize the network parameters by sampling from a gaussian distribution. We also set appropriate parameters like regularization coefficient, dropout probabilities, learning rate, etc. as shown below. More parameters can be found in the documentation of :sgclass:`NeuralNetwork`.

.. sgexample:: convolutional_net_classification.sg:set_parameters

.. sgexample:: convolutional_net_classification.sg:create_instance

We create instances of :sgclass:`NeuralInputLayer`, :sgclass:`NeuralConvolutionalLayer` and :sgclass:`NeuralSoftmaxLayer` which are building blocks of :sgclass:`NeuralNetwork`

.. sgexample:: convolutional_net_classification.sg:add_layers

We train the model and apply it to some test data.

.. sgexample:: convolutional_net_classification.sg:train_and_apply

Finally, we compute accuracy.

.. sgexample:: convolutional_net_classification.sg:evaluate_accuracy

----------
References
----------
:wiki:`Convolutional_neural_network`

.. bibliography:: ../../references.bib
    :filter: docname in docnames
