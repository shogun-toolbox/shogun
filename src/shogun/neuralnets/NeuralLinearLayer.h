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

	/** Initializes the layer, computes the number of parameters needed for
	 * the layer
	 *
	 * @param layers Array of layers that form the network that this layer is
	 * being used with
	 *
	 * @param input_indices  Indices of the layers that are connected to this
	 * layer as input
	 */
	virtual void initialize(CDynamicObjectArray* layers,
			SGVector<int32_t> input_indices);

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

	/** Computes
	 * \f[ \frac{\lambda}{N} \sum_{k=0}^{N-1} \left \| J(x_k) \right \|^2_F \f]
	 * where \f$ \left \| J(x_k)) \right \|^2_F \f$ is the Frobenius norm of
	 * the Jacobian of the activations of the hidden layer with respect to its
	 * inputs, \f$ N \f$ is the batch size, and \f$ \lambda \f$ is the
	 * contraction coefficient.
	 *
	 * Should be implemented by layers that support being used as a hidden
	 * layer in a contractive autoencoder.
	 *
	 * @param parameters Vector of size get_num_parameters(), contains the
	 * parameters of the layer
	 */
	virtual float64_t compute_contraction_term(SGVector<float64_t> parameters);

	/** Adds the gradients of
	 * \f[ \frac{\lambda}{N} \sum_{k=0}^{N-1} \left \| J(x_k) \right \|^2_F \f]
	 * to the gradients vector, where \f$ \left \| J(x_k)) \right \|^2_F \f$ is
	 * the Frobenius norm of the Jacobian of the activations of the hidden layer
	 * with respect to its inputs, \f$ N \f$ is the batch size, and
	 * \f$ \lambda \f$ is the contraction coefficient.
	 *
	 * Should be implemented by layers that support being used as a hidden
	 * layer in a contractive autoencoder.
	 *
	 * @param parameters Vector of size get_num_parameters(), contains the
	 * parameters of the layer
	 * @param gradients Vector of size get_num_parameters(). Gradients of the
	 * contraction term will be added to it
	 */
	virtual void compute_contraction_term_gradients(
		SGVector<float64_t> parameters, SGVector<float64_t> gradients);

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

	virtual const char* get_name() const { return "NeuralLinearLayer"; }
};

}
#endif
