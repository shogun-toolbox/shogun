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

CNeuralRecurrentLayer::CNeuralRecurrentLayer(int32_t num_neurons, int32_t time_series_length, int32_t output_dim):
CNeuralLinearLayer(num_neurons)
{
	m_time_series_length = time_series_length;
 	m_output_dim = output_dim;
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
		// V: hidden to output weights, shape [m_output_dim, m_num_neurons]
  	m_num_parameters += m_output_dim * m_num_neurons;
	}

 // Initialize all hidden states
	m_hidden_states = std::vector<SGMatrix<float64_t>>(m_time_series_length);
	for (int i = 0; i < m_time_series_length; i++) {
		m_hidden_states[i] = SGMatrix<float64_t>(m_num_neurons, m_num_neurons);
	}
}

void CNeuralRecurrentLayer::set_batch_size(int32_t batch_size)
{
	m_batch_size = batch_size;

	m_activations = SGMatrix<float64_t>(m_output_dim, m_batch_size);
	m_hidden_activation = SGMatrix<float64_t>(m_num_neurons, m_batch_size);

	// TODO implement dropout
	// m_dropout_mask = SGMatrix<bool>(m_num_neurons, m_batch_size);

	m_outputs = std::vector<SGMatrix<float64_t>>(m_time_series_length);
	for (int i = 0; i < m_time_series_length; i++) {
		m_outputs[i] = SGMatrix<float64_t>(m_output_dim, batch_size);
	}
	// TODO finish this
	//if (!is_input())
	//{
	//	m_activation_gradients =
	//		SGMatrix<float64_t>(m_num_neurons, m_batch_size);
	//	m_local_gradients = SGMatrix<float64_t>(m_num_neurons, m_batch_size);
	//}
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
// VERY BIG TODO: Figure out how to manage input from multiple layers
// Now we assume that we only have 1 input layer

void CNeuralRecurrentLayer::compute_activations(SGVector<float64_t> parameters,
		CDynamicObjectArray* layers)
{
	float64_t* hidden_biases = parameters.vector;
	float64_t* output_biases = parameters.vector + m_num_neurons;

	typedef Eigen::Map<Eigen::MatrixXd> EMappedMatrix;
	typedef Eigen::Map<Eigen::VectorXd> EMappedVector;

	EMappedMatrix  A(m_activations.matrix, m_output_dim, m_batch_size);
	EMappedMatrix  H(m_hidden_activation.matrix, m_num_neurons, m_batch_size);
	EMappedVector  Bh(hidden_biases, m_num_neurons);
	EMappedVector  By(output_biases, m_num_neurons);

	int32_t weights_index_offset = m_num_neurons * 2;

	for (int i = 0; i < m_time_series_length; i++) {

		float64_t* weights = parameters.vector + weights_index_offset;

		weights_index_offset += m_num_neurons * m_input_sizes[0];
		float64_t* hidden_weights = parameters.vector + weights_index_offset;

		weights_index_offset += m_num_neurons * m_num_neurons;
		float64_t* output_weights = parameters.vector + weights_index_offset;
		weights_index_offset += m_num_neurons * m_input_sizes[0];

		EMappedMatrix Wxh(weights, m_num_neurons, m_input_sizes[0]);
		EMappedMatrix Whh(hidden_weights, m_num_neurons, m_num_neurons);
		EMappedMatrix Why(output_weights, m_input_sizes[0], m_num_neurons);

		CNeuralLayer* layer =
			(CNeuralLayer*)layers->element(m_input_indices[0]);

		EMappedMatrix X(layer->get_activations().matrix,
				layer->get_num_neurons(), m_batch_size);

		// TODO Maybe do something to be able to choose ReLU instead,
		// but ReLU kill too much gradient when unrolling the recursion
		// in backprop
		H += (Wxh * X + Whh * H).colwise() + Bh;
		H = H.unaryExpr<float64_t(*)(float64_t)>(&std::tanh);
		// Using copy constructor
		m_hidden_states[i] = SGMatrix<float64_t>(&m_hidden_activation);
		A += (Why * H).colwise() + By;
		// Using copy constructor
		m_outputs[i] = SGMatrix<float64_t>(&m_activations);
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
	for (int32_t i = 0; i < m_time_series_length; i++) {
		SG_ADD(&m_hidden_states[i], "hidden_states",
		       "Hidden States", MS_NOT_AVAILABLE);
	}
}
