/*
 * Copyright (c) 2017, Shogun Toolbox Foundation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Written (W) 2017 Olivier Nguyen
 */

#ifndef __NEURALRECURRENTLAYER_H__
#define __NEURALRECURRENTLAYER_H__

#include <shogun/lib/common.h>
#include <shogun/neuralnets/NeuralLinearLayer.h>

namespace shogun
{
/** @brief Recurrent Neural layer with linear neurons, with an identity activation
 * function. can be used as a hidden layer or an output layer
 *
 * Each neuron in the layer is connected to all the neurons in all the input 
 * and hidden layers that connect into this layer.
 *
 * The hidden state at time step t is computed according to the previous hidden state
 * and the input at the current state as:
 *
 * \f$ h_t = f(W x_t + U s_t_-_1) + b_h \f$ where \f$ b_h \f$ is the bias vector of 
 * the hidden layer, \f$ W \f$ is the weights matrix between this layer and the input layer, 
 * and \f$ U \f$ is the weights matrix between this hidden layer and itself
 *
 * The output layer is computed as:
 *
 * \f$ y_t = f(V h_t) + b_y \f$ where \f$ b_y \f$ is the bias vector of the output layer, 
 * \f$ h_t \f$ is the value of the hidden states at the current time step and \f$ V \f$ is 
 * the weights matrix between this hidden layer and the output layer
 *
 */
class CNeuralRecurrentLayer : public CNeuralLinearLayer
{
public:
	/** default constructor */
	CNeuralRecurrentLayer();

	/** Constuctor
	 *
	 * @param num_neurons Number of neurons in this layer
	 */
	CNeuralRecurrentLayer(int32_t num_neurons);

	virtual ~CNeuralRecurrentLayer() {}

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

	/** Sets the batch_size and allocates memory for m_activations and
	 * m_input_gradients accordingly. Must be called before forward or backward
	 * propagation is performed
	 *
	 * @param batch_size number of training/test cases the network is
	 * currently working with
	 */
	virtual void set_batch_size(int32_t batch_size);

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
	virtual void initialize_parameters(SGVector<float64_t> parameters,
			SGVector<bool> parameter_regularizable,
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
	virtual void compute_activations(SGVector<float64_t> parameters,
			CDynamicObjectArray* layers);

	/** Computes the gradients that are relevent to this layer:
	 *- The gradients of the error with respect to the layer's parameters
	 * -The gradients of the error with respect to the layer's inputs
	 *
	 * Input gradients for layer i that connects into this layer as input are
	 * added to m_layers.element(i).get_activation_gradients()
	 *
	 * Deriving classes should make sure to account for
	 * [dropout](http://arxiv.org/abs/1207.0580) [Hinton, 2012] during gradient
	 * computations
	 *
	 * @param parameters Vector of size get_num_parameters(), contains the
	 * parameters of the layer
	 *
	 * @param targets a matrix of size num_neurons*batch_size. If the layer is
	 * being used as an output layer, targets is the desired values for the
	 * layer's activations, otherwise it's an empty matrix
	 *
	 * @param layers Array of layers that form the network that this layer is
	 * being used with
	 *
	 * @param parameter_gradients Vector of size get_num_parameters(). To be
	 * filled with gradients of the error with respect to each parameter of the
	 * layer
	 */
	virtual void compute_gradients(SGVector<float64_t> parameters,
			SGMatrix<float64_t> targets,
			CDynamicObjectArray* layers,
			SGVector<float64_t> parameter_gradients);

	/** Computes the error between the layer's current activations and the given
	 * target activations. Should only be used with output layers
	 *
	 * @param targets desired values for the layer's activations, matrix of size
	 * num_neurons*batch_size
	 */
	virtual float64_t compute_error(SGMatrix<float64_t> targets);

	/** Gets the layer's hidden states, a matrix of size num_neurons * batch_size
	 *
	 * @return layer's activations
	 */
	virtual SGMatrix<float64_t> get_hidden_states() { return m_hidden_states; }

	/** Computes the gradients of the error with respect to this layer's
	 * pre-activations. Results are stored in m_local_gradients.
	 *
	 * This is used by compute_gradients() and can be overriden to implement
	 * layers with different activation functions
	 *
	 * @param targets a matrix of size num_neurons*batch_size. If the layer is
	 * being used as an output layer, targets is the desired values for the
	 * layer's activations, otherwise it's an empty matrix
	 */
	virtual void compute_local_gradients(SGMatrix<float64_t> targets);

	virtual const char* get_name() const { return "NeuralRecurrentLayer"; }

private:
	/** initializes the hidden states of this layer
	 */
	void init();

protected:

	/** hidden states in this layer
	 * size num_neurons * batch_size
	 */
	SGMatrix<float64_t> m_hidden_states;

};

}
#endif
