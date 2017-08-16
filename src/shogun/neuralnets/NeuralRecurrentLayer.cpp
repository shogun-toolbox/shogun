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

#include <shogun/neuralnets/NeuralRecurrentLayer.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGVector.h>

#include <shogun/mathematics/eigen3.h>

using namespace shogun;

CNeuralRecurrentLayer::CNeuralRecurrentLayer() : CNeuralLinearLayer()
{
}

CNeuralRecurrentLayer::CNeuralRecurrentLayer(int32_t num_neurons):
CNeuralLinearLayer(num_neurons)
{
}

void CNeuralRecurrentLayer::initialize_neural_layer(CDynamicObjectArray* layers,
		SGVector< int32_t > input_indices)
{
	CNeuralLayer::initialize_neural_layer(layers, input_indices);

	// hidden bias + output bias
	m_num_parameters += m_num_neurons * 2;
	for (int32_t i = 0; i < input_indices.vlen; i++)
	{
		// W: input to hidden weights, shape [m_num_neurons, m_input_size]
		m_num_parameters += m_num_neurons * m_input_sizes[i];
		// U: hidden to hidden weights, shape [m_num_neurons, m_num_neurons]
		m_num_parameters += m_num_neurons * m_num_neurons;
		// V: hidden to output weights, shape [m_input_size, m_num_neurons]
		m_num_parameters += m_input_sizes[i] * m_num_neurons;
	}
}

void CNeuralRecurrentLayer::set_batch_size(int32_t batch_size)
{
	CNeuralLayer::set_batch_size(batch_size);
	m_hidden_states = SGMatrix<float64_t>(m_num_neurons, m_batch_size);
}

void CNeuralRecurrentLayer::initialize_parameters(SGVector<float64_t> parameters,
		SGVector<bool> parameter_regularizable,
		float64_t sigma)
{
	for (int32_t i=0; i<m_num_parameters; i++)
	{
		// random the parameters
		parameters[i] = CMath::normal_random(0.0, sigma);

		// turn regularization off for the biases, on for the weights
		// parameter_regularizable[i] = (i >= m_num_neurons);
	}
}

void CNeuralRecurrentLayer::compute_activations(SGVector<float64_t> parameters,
		CDynamicObjectArray* layers)
{
	float64_t* hidden_biases = parameters.vector;
	float64_t* output_biases = parameters.vector + m_num_neurons;

	typedef Eigen::Map<Eigen::MatrixXd> EMappedMatrix;
	typedef Eigen::Map<Eigen::VectorXd> EMappedVector;

	EMappedMatrix  A(m_activations.matrix, m_num_neurons, m_batch_size);
	EMappedMatrix  H(m_hidden_states.matrix, m_num_neurons, m_batch_size);
	EMappedVector  Bh(hidden_biases, m_num_neurons);
	EMappedVector  By(output_biases, m_num_neurons);

	A.colwise() = By;
	H.colwise() = Bh;

	int32_t weights_index_offset = m_num_neurons * 2;
	for (int32_t l=0; l<m_input_indices.vlen; l++)
	{
		CNeuralLayer* layer =
			(CNeuralLayer*)layers->element(m_input_indices[l]);

		float64_t* weights = parameters.vector + weights_index_offset;

		weights_index_offset += m_num_neurons * m_input_sizes[l];
		float64_t* hidden_weights = parameters.vector + weights_index_offset;

		weights_index_offset += m_num_neurons * m_num_neurons;
		float64_t* output_weights = parameters.vector + weights_index_offset;
		weights_index_offset += m_num_neurons * m_input_sizes[l];

		EMappedMatrix W(weights, m_num_neurons, m_input_sizes[l]);
		EMappedMatrix Wxh(weights, m_num_neurons, m_input_sizes[l]);
		EMappedMatrix Whh(hidden_weights, m_num_neurons, m_num_neurons);
		EMappedMatrix Why(output_weights, m_input_sizes[l], m_num_neurons);
		EMappedMatrix X(layer->get_activations().matrix,
				layer->get_num_neurons(), m_batch_size);

		// TODO: Need to pass through activation function
		// Either tanh or ReLU
		// H += Wxh * X + Whh * H;
		// A += Why * H;
		SG_UNREF(layer);
	}
}

void CNeuralRecurrentLayer::compute_gradients(
		SGVector<float64_t> parameters,
		SGMatrix<float64_t> targets,
		CDynamicObjectArray* layers,
		SGVector<float64_t> parameter_gradients)
{
}

void CNeuralRecurrentLayer::compute_local_gradients(SGMatrix<float64_t> targets)
{
}

float64_t CNeuralRecurrentLayer::compute_error(SGMatrix<float64_t> targets)
{
	return 0.0;
}

void CNeuralRecurrentLayer::init()
{
	SG_ADD(&m_hidden_states, "hidden_states",
	       "Hidden States", MS_NOT_AVAILABLE);
}

