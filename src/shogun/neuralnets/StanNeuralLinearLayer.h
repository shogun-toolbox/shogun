/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk, Khaled Nasr
 */

#ifndef __STANNEURALLINEARLAYER_H__
#define __STANNEURALLINEARLAYER_H__

#include <shogun/neuralnets/StanNeuralLayer.h>
#include <shogun/lib/common.h>

namespace shogun
{
/** @brief Neural layer with linear neurons, with an identity activation
 * function. can be used as a hidden layer or an output layer
 *
 * Each neuron in the layer is connected to all the neurons in all the layers
 * that connect into this layer.
 *
 * Activations for each train/test case are computed according to
 * \f$ b + \sum_i W_i x_i \f$ where \f$ b \f$ is the bias vector, \f$ W_i \f$
 * is the weights matrix between this layer and layer i of its inputs, and
 * \f$ x_i \f$ is the activations vector of layer i.
 *
 * The layout of the parameter vector of this layer is as follows:
 * 	- The first num_neurons elements correspond to the biases
 * 	- The following elements correspond to the weight matrices. For each layer
 * i that connects into this layer as input, a weight matrix of size
 * num_neurons*num_neurons_i is stored in column major format.
 *
 * When used as an output layer, a
 * [squared error measure](http://en.wikipedia.org/wiki/Mean_squared_error) is
 * used
 */
class StanNeuralLinearLayer : public StanNeuralLayer
{
public:
	/** default constructor */
	StanNeuralLinearLayer();

	/** Constuctor
	 *
	 * @param num_neurons Number of neurons in this layer
	 */
	StanNeuralLinearLayer(int32_t num_neurons);

	virtual ~StanNeuralLinearLayer() {}

	/** Initializes the layer, computes the number of parameters needed for
	 * the layer
	 *
	 * @param layers Array of layers that form the network that this layer is
	 * being used with
	 *
	 * @param input_indices  Indices of the layers that are connected to this
	 * layer as input
	 */
	virtual void initialize_neural_layer(CDynamicObjectArray* layers,
			SGVector<int32_t> input_indices);

	/** Initializes the layer's parameters. The layer should fill the given
	 * arrays with the initial value for its parameters
	 *
	 * @param parameters Vector of size get_num_parameters()
	 *
	 * @param sigma standard deviation of the gaussian used to random the
	 * parameters
	 */
	virtual void initialize_parameters(StanVector& parameters, int32_t i, int32_t j,
			float64_t sigma);

	/** Computes the activations of the neurons in this layer, results should
	 * be stored in m_activations. To be used only with non-input layers
	 *
	 * @param parameters Vector of size get_num_parameters(), contains the
	 * parameters of the layer
	 *
	 * @param layers Array of layers that form the network that this layer is
	 * being used with
	 */
	virtual void compute_activations(StanVector& parameters, int32_t, int32_t j,
			CDynamicObjectArray* layers);

	virtual const char* get_name() const { return "StanNeuralLinearLayer"; }
};

}
#endif
