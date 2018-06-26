/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk, Khaled Nasr
 */

#ifndef __STANNEURALLAYER_H__
#define __STANNEURALLAYER_H__

#include <shogun/optimization/StanFirstOrderSAGCostFunction.h>
#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/DynamicObjectArray.h>

namespace shogun
{

template <class T> class SGVector;

/** @brief Base class for neural network layers
 *
 * A Neural layer represents an group of neurons and is the basic building block
 * for neural networks.
 *
 * This class is to be inherited from to implement layers with different neuron
 * types (i.e linear, softmax, convolutional, etc..)
 *
 * Any arbitrary layer type can be derived from this class provided that the
 * following functions have been defined in a mathematically plausible manner:
 * - initialize_parameters()
 * - compute_activations()
 * - compute_gradients()
 * - compute_error() [only if the layer can be used as an output layer]
 * - enforce_max_norm()
 *
 * The memory for the layer's parameters (weights and biases) is not allocated
 * by the layer itself, instead it is allocated by the network that the layer
 * belongs to, and passed to the layer when it needs to use it.
 *
 * This class stores buffers for use during forward and backpropagation, this is
 * to avoid unnecessary memory allocation during computations. The buffers are:
 * m_activations: size m_num_neurons*m_batch_size
 * m_activation_gradients: size m_num_neurons*m_batch_size
 * m_local_gradients: size m_num_neurons*m_batch_size
 */
class StanNeuralLayer : public CSGObject
{
public:
	/** default constructor */
	StanNeuralLayer();

	/** Constuctor
	 *
	 * @param num_neurons Number of neurons in this layer
	 */
	StanNeuralLayer(int32_t num_neurons);

	virtual ~StanNeuralLayer();

	/** Initializes the layer
	 *
	 * @param layers Array of layers that form the network that this layer is
	 * being used with
	 *
	 * @param input_indices  Indices of the layers that are connected to this
	 * layer as input
	 */
	virtual void initialize_neural_layer(CDynamicObjectArray* layers,
			SGVector<int32_t> input_indices);

	/** Sets the batch_size and allocates memory for m_activations and
	 * m_input_gradients accordingly. Must be called before forward or backward
	 * propagation is performed
	 *
	 * @param batch_size number of training/test cases the network is
	 * currently working with
	 */
	virtual void set_batch_size(int32_t batch_size);

	/** returns true if the layer is an input layer. Input layers are the root
	 * layers of a network, that is, they don't receive signals from other
	 * layers, they receive signals from the inputs features to the network.
	 *
	 * Local and activation gradients are not computed for input layers
	 */
	virtual bool is_input() { return false; }

	/** Initializes the layer's parameters. The layer should fill the given
	 * arrays with the initial value for its parameters
	 *
	 * @param parameters Vector of size get_num_parameters()
	 *
	 * @param parameter_regularizable Vector of size get_num_parameters().
	 * This controls which of the layer's parameter are
	 * subject to regularization, i.e to turn off regularization for parameter
	 * i, set parameter_regularizable[i] = false. This is usally used to turn
	 * off regularization for bias parameters.
	 *
	 * @param sigma standard deviation of the gaussian used to random the
	 * parameters
	 */
	virtual void initialize_parameters(StanVector& parameters,
			int32_t i, int32_t j,
			float64_t sigma) { }

	/** Computes the activations of the neurons in this layer, results should
	 * be stored in m_activations. To be used only with input layers
	 *
	 * @param inputs activations of the neurons in the
	 * previous layer, matrix of size previous_layer_num_neurons * batch_size
	 */
	virtual void compute_activations(StanMatrix inputs) { }

	/** Computes the activations of the neurons in this layer, results should
	 * be stored in m_activations. To be used only with non-input layers
	 *
	 * @param parameters Vector of size get_num_parameters(), contains the
	 * parameters of the layer
	 *
	 * @param layers Array of layers that form the network that this layer is
	 * being used with
	 */
	virtual void compute_activations(StanVector& parameters, int32_t i, int32_t j,
			CDynamicObjectArray* layers) { }

	/** Applies [dropout](http://arxiv.org/abs/1207.0580) [Hinton, 2012] to the
	 * activations of the layer
	 *
	 * If is_training is true, fills m_dropout_mask with random values
	 * (according to dropout_prop) and multiplies it into the activations,
	 * otherwise, multiplies the activations by (1-dropout_prop) to compensate
	 * for using dropout during training
	 */
	virtual void dropout_activations();

	/** Gets the number of neurons in the layer
	 *
	 * @return number of neurons in the layer
	 */
	virtual int32_t get_num_neurons() { return m_num_neurons; }

	/** Returns the width assuming that the layer's activations are interpreted as
	 * images (i.e for convolutional nets)
	 *
	 * @return Width
	 */
	virtual int32_t get_width() { return m_width; }

	/** Returns the height assuming that the layer's activations are interpreted as
	 * images (i.e for convolutional nets)
	 *
	 * @return Height
	 */
	virtual int32_t get_height() { return m_height; }

	/** Gets the number of neurons in the layer
	 *
	 * @param num_neurons number of neurons in the layer
	 */
	virtual void set_num_neurons(int32_t num_neurons)
	{
		m_num_neurons = num_neurons;
		set_batch_size(m_batch_size);
	}

	/** Gets the number of parameters used in this layer
	 *
	 * @return number of parameters used in this layer
	 */
	virtual int32_t get_num_parameters() { return m_num_parameters; }

	/** Gets the layer's activations, a matrix of size num_neurons * batch_size
	 *
	 * @return layer's activations
	 */
	virtual StanMatrix get_activations() { return m_stan_activations; }

	/** Gets the indices of the layers that are connected to this layer as input
	 *
	 * @return layer's input indices
	 */
	virtual SGVector<int32_t> get_input_indices() { return m_input_indices; }

	virtual const char* get_name() const { return "NeuralLayer"; }

private:
	void init();

public:
	/** Should be true if the layer is currently used during training
	 * initial value is false
	 */
	bool is_training;

	/** probabilty of dropping out a neuron in the layer */
	float64_t dropout_prop;

protected:
	/** Number of neurons in this layer */
	int32_t m_num_neurons;

	/** Width of the image (if the layer's activations are to be interpreted as
	 * images. Default value is m_num_neurons
	 */
	int32_t m_width;

	/** Width of the image (if the layer's activations are to be interpreted as
	 * images. Default value is 1
	 */
	int32_t m_height;

	/** Number of neurons in this layer */
	int32_t m_num_parameters;

	/** Indices of the layers that are connected to this layer as input */
	SGVector<int32_t> m_input_indices;

	/** Number of neurons in the layers that are connected to this layer as
	 * input
	 */
	SGVector<int32_t> m_input_sizes;

	/** number of training/test cases the network is currently working with */
	int32_t m_batch_size;

  /** activations of the neurons in this layer as stan Matrix
   *  size num_neurons * batch_size
   */
  StanMatrix m_stan_activations;

	/** binary mask that determines whether a neuron will be kept or dropped out
	 * during the current iteration of training
	 * size num_neurons * batch_size
	 */
	SGMatrix<bool> m_dropout_mask;
};

}
#endif //__STANNEURALLAYER_H__
