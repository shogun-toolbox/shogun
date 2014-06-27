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

#ifndef __NEURALLOGISTICLAYER_H__
#define __NEURALLOGISTICLAYER_H__

#include <shogun/neuralnets/NeuralLinearLayer.h>

namespace shogun
{
/** @brief Neural layer with linear neurons, with a [logistic activation 
 * function](http://en.wikipedia.org/wiki/Logistic_function). can be used as a 
 * hidden layer or an output layer.
 * 
 * When used as an output layer, a 
 * [squared error measure](http://en.wikipedia.org/wiki/Mean_squared_error) is 
 * used
 */
class CNeuralLogisticLayer : public CNeuralLinearLayer
{
public:
	/** default constructor */
	CNeuralLogisticLayer();
	
	/** Constuctor
	 * 
	 * @param num_neurons Number of neurons in this layer
	 */
	CNeuralLogisticLayer(int32_t num_neurons);
	
	virtual ~CNeuralLogisticLayer() {}
	
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
	
	virtual const char* get_name() const { return "NeuralLogisticLayer"; }
};
	
}
#endif
