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

#ifndef __NEURALRECURRENTNETWORK_H__
#define __NEURALRECURRENTNETWORK_H__

#include <shogun/lib/common.h>
#include <shogun/machine/Machine.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/neuralnets/NeuralNetwork.h>

namespace shogun
{
template<class T> class CDenseFeatures;
class CDynamicObjectArray;
class CNeuralLayer;
class CNeuralRecurrentLayer;

/** @brief A recurrent neural network
 *
 */
class CNeuralRecurrentNetwork : public CNeuralNetwork
{

public:
	/** default constuctor */
	CNeuralRecurrentNetwork();

	/** Sets the layers of the network
	 *
	 * @param layers An array of CNeuralLayer objects specifying the layers of
	 * the network. Must contain at least one input layer. The last layer in
	 * the array is treated as the output layer
	 */
	CNeuralRecurrentNetwork(CDynamicObjectArray* layers);

	/** Initializes the network
	 *
	 * @param sigma standard deviation of the gaussian used to randomly
	 * initialize the parameters
	 */
	virtual void initialize_neural_network(float64_t sigma = 0.01f);

	virtual ~CNeuralRecurrentNetwork();

	/** apply machine to data in means of binary classification problem */
	virtual CBinaryLabels* apply_binary(CFeatures* data);
	/** apply machine to data in means of regression problem */
	virtual CRegressionLabels* apply_regression(CFeatures* data);
	/** apply machine to data in means of multiclass classification problem */
	virtual CMulticlassLabels* apply_multiclass(CFeatures* data);

	/** get classifier type
	 *
	 * @return classifier type CT_NEURALNETWORK
	 */
	virtual EMachineType get_classifier_type() { return CT_NEURALNETWORK; }

	virtual const char* get_name() const { return "NeuralRecurrentNetwork";}

protected:
	/** trains the network */
	virtual bool train_machine(CFeatures* data=NULL);

	/** trains the network using gradient descent*/
	virtual bool train_gradient_descent(SGMatrix<float64_t> inputs,
			SGMatrix<float64_t> targets);

	/** Applies forward propagation, computes the activations of each layer up
	 * to layer j
	 *
	 * @param data input features
	 * @param j layer index at which the propagation should stop. If -1, the
	 * propagation continues up to the last layer
	 *
	 * @return activations of the last layer
	 */
	virtual SGMatrix<float64_t> forward_propagate(CFeatures* data, int32_t j=-1);

	/** Applies forward propagation, computes the activations of each layer up
	 * to layer j
	 *
	 * @param inputs inputs to the network, a matrix of size
	 * m_num_inputs*m_batch_size
	 * @param j layer index at which the propagation should stop. If -1, the
	 * propagation continues up to the last layer
	 *
	 * @return activations of the last layer
	 */
	virtual SGMatrix<float64_t> forward_propagate(SGMatrix<float64_t> inputs, int32_t j=-1);

	/** Applies backpropagation to compute the gradients of the error with
	 * repsect to every parameter in the network.
	 *
	 * @param inputs inputs to the network, a matrix of size
	 * m_num_inputs*m_batch_size
	 *
	 * @param targets desired values for the output layer's activations. matrix
	 * of size m_layers[m_num_layers-1].get_num_neurons()*m_batch_size
	 *
	 * @param gradients array to be filled with gradient values.
	 *
	 * @return error between the targets and the activations of the last layer
	 */
	virtual float64_t compute_gradients(SGMatrix<float64_t> inputs,
			SGMatrix<float64_t> targets, SGVector<float64_t> gradients);

	/** Forward propagates the inputs and computes the error between the output
	 * layer's activations and the given target activations.
	 *
	 * @param inputs inputs to the network, a matrix of size
	 * m_num_inputs*m_batch_size
	 *
	 * @param targets desired values for the network's output, matrix of size
	 * num_neurons_output_layer*batch_size
	 */
	virtual float64_t compute_error(SGMatrix<float64_t> inputs,
			SGMatrix<float64_t> targets);

	/** Computes the error between the output layer's activations and the given
	 * target activations.
	 *
	 * @param targets desired values for the network's output, matrix of size
	 * num_neurons_output_layer*batch_size
	 */
	virtual float64_t compute_error(SGMatrix<float64_t> targets);

private:
	void init();

protected:
};

}
#endif
