/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk, Khaled Nasr
 */

#include <shogun/neuralnets/StanNeuralLinearLayer.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGVector.h>

#include <shogun/mathematics/eigen3.h>

using namespace shogun;

StanNeuralLinearLayer::StanNeuralLinearLayer() : StanNeuralLayer()
{
}

StanNeuralLinearLayer::StanNeuralLinearLayer(int32_t num_neurons):
StanNeuralLayer(num_neurons)
{
}

//Updated
void StanNeuralLinearLayer::initialize_neural_layer(CDynamicObjectArray* layers,
		SGVector< int32_t > input_indices)
{
	StanNeuralLayer::initialize_neural_layer(layers, input_indices);

	m_num_parameters = m_num_neurons;
	for (int32_t i=0; i<input_indices.vlen; i++)
		m_num_parameters += m_num_neurons*m_input_sizes[i];
}

void StanNeuralLinearLayer::initialize_parameters(StanVector& parameters, int32_t i, int32_t j,
		float64_t sigma)
{
	for (int32_t idx=i; idx<= j; idx++)
	{
		// random the parameters
		parameters(idx,0) = CMath::normal_random(0.0, sigma);
	}
}

void StanNeuralLinearLayer::compute_activations(StanVector& parameters, int32_t i, int32_t j,
		CDynamicObjectArray* layers)
{
	auto biases = parameters.block(i,0,m_num_neurons, 1);
	StanMatrix& A = m_stan_activations;
	A.resize(m_num_neurons, m_batch_size);
	A.colwise() = biases;

	int32_t weights_index_offset = m_num_neurons;
	for (int32_t l=0; l<m_input_indices.vlen; l++)
	{
		StanNeuralLayer* layer =
			(StanNeuralLayer*)layers->element(m_input_indices[l]);

		//float64_t* weights = parameters.vector + weights_index_offset;
		auto W = parameters.block(i+weights_index_offset, 0, m_num_neurons*layer->get_num_neurons(), 1);
		W.resize(m_num_neurons, layer->get_num_neurons());
		weights_index_offset += m_num_neurons*layer->get_num_neurons();

		//EMappedMatrix W(weights, m_num_neurons, layer->get_num_neurons());
		auto X = layer->get_activations();
		X.resize(layer->get_num_neurons(), m_batch_size);

		A += W*X;
		SG_UNREF(layer);
	}
}
