/*
 * Copyright (c) 2014, Shogun Toolbox Foundation
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
 * Written (W) 2014 Khaled Nasr
 */

#ifndef __NEURALLINEARLAYER_H__
#define __NEURALLINEARLAYER_H__

#include <shogun/lib/common.h>
#include <shogun/neuralnets/NeuralLayer.h>

namespace shogun
{
/** @brief Neural layer with linear neurons, with an identity activation 
 * function. can be used as a hidden layer or an output layer
 * 
 * Each neuron in the layer is connected to all the neurons in the previous
 * layer
 * 
 * When used as an output layer, a 
 * [squared error measure](http://en.wikipedia.org/wiki/Mean_squared_error) is 
 * used
 */
class CNeuralLinearLayer : public CNeuralLayer
{
public:
	/** default constructor */
	CNeuralLinearLayer();
	
	/** Constuctor
	 * 
	 * @param num_neurons Number of neurons in this layer
	 */
	CNeuralLinearLayer(int32_t num_neurons);
	
	virtual ~CNeuralLinearLayer() {}
	
	/** Gets the number of parameters (weights and biases) needed for this 
	 * layer
	 * 
	 * @return number of parameters (weights and biases) needed for this layer
	 */
	virtual int32_t get_num_parameters();
	
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
	 * be stored in m_activations
	 * 
	 * @param parameters Vector of size get_num_parameters(), contains the 
	 * parameters of the layer
	 * 
	 * @param previous_layer_activations activations of the neurons in the 
	 * previous layer, matrix of size previous_layer_num_neurons * batch_size
	 */
	virtual void compute_activations(SGVector<float64_t> parameters,
			SGMatrix<float64_t> previous_layer_activations);
	
	/** Computes the gradients that are relevent to this layer:
	 *- The gradients of the error with respect to the layer's parameters
	 * -The gradients of the error with respect to the layer's inputs
	 * 
	 * Deriving classes should make sure to account for 
	 * [dropout](http://arxiv.org/abs/1207.0580) [Hinton, 2012] during gradient 
	 * computations
	 * 
	 * The input gradients are stored in m_input_gradients
	 *
	 * @param parameters Vector of size get_num_parameters(), contains the 
	 * parameters of the layer
	 * 
	 * @param is_output specifies if the layer is used as an output layer or a
	 * hidden layer
	 * 
	 * @param p a matrix of size num_neurons*batch_size. If is_output is true,p 
	 * is the desired values for the layer's activations, else it is the
	 * gradients of the error with respect to this layer's activations (the 
	 * input gradients of the next layer).
	 *
	 * @param previous_layer_activations activations of the neurons in the
	 * previous layer, matrix of size previous_layer_num_neurons * batch_size
	 * 
	 * @param parameter_gradients Vector of size get_num_parameters(). To be 
	 * filled with gradients of the error with respect to each parameter of the 
	 * layer
	 */
	virtual void compute_gradients(SGVector<float64_t> parameters, 
			bool is_output,
			SGMatrix<float64_t> p,
			SGMatrix<float64_t> previous_layer_activations,
			SGVector<float64_t> parameter_gradients);
	
	/** Computes the error between the layer's current activations and the given
	 * target activations. Should only be used with output layers
	 *
	 * @param targets desired values for the layer's activations, matrix of size
	 * num_neurons*batch_size
	 */
	virtual float64_t compute_error(SGMatrix<float64_t> targets);
	
	/** Constrains the weights of each neuron in the layer to have an L2 norm of
	 * at most max_norm
	 * 
	 * @param parameters pointer to the layer's parameters, array of size 
	 * get_num_parameters()
	 * 
	 * @param max_norm maximum allowable norm for a neuron's weights
	 */
	virtual void enforce_max_norm(SGVector<float64_t> parameters, 
			float64_t max_norm);
	
	/** Computes the gradients of the error with respect to this layer's
	 * activations. Results are stored in m_local_gradients. 
	 * 
	 * This is used by compute_gradients() and can be overriden to implement 
	 * layers with different activation functions
	 *
	 * @param is_output specifies if the layer is used as an output layer or a
	 * hidden layer
	 * 
	 * @param p a matrix of size num_neurons*batch_size. If is_output is true,p 
	 * is the desired values for the layer's activations, else it is the
	 * gradients of the error with respect to this layer's activations (the 
	 * input gradients of the next layer).
	 */
	virtual void compute_local_gradients(bool is_output, SGMatrix<float64_t> p);
	
	virtual const char* get_name() const { return "NeuralLinearLayer"; }
};
	
}
#endif
