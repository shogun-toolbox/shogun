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

#include <shogun/neuralnets/Autoencoder.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralLinearLayer.h>
#include <shogun/neuralnets/NeuralConvolutionalLayer.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

Autoencoder::Autoencoder() : NeuralNetwork()
{
	init();
}

Autoencoder::Autoencoder(int32_t num_inputs, const std::shared_ptr<NeuralLayer>& hidden_layer,
	std::shared_ptr<NeuralLayer> decoding_layer, float64_t sigma) : NeuralNetwork()
{
	init();

	if (decoding_layer==NULL)
		decoding_layer = std::make_shared<NeuralLinearLayer>(num_inputs);

	std::vector<std::shared_ptr<NeuralLayer>> layers;
	layers.push_back(std::make_shared<NeuralInputLayer>(num_inputs));
	layers.push_back(hidden_layer);
	layers.push_back(decoding_layer);

	set_layers(layers);

	quick_connect();

	hidden_layer->autoencoder_position = NLAP_ENCODING;
	decoding_layer->autoencoder_position = NLAP_DECODING;

	initialize_neural_network(sigma);
}

Autoencoder::Autoencoder(
	int32_t input_width, int32_t input_height, int32_t input_num_channels,
	const std::shared_ptr<NeuralConvolutionalLayer>& hidden_layer,
	const std::shared_ptr<NeuralConvolutionalLayer>& decoding_layer,
	float64_t sigma)
	: NeuralNetwork()
{
	init();

	std::vector<std::shared_ptr<NeuralLayer>> layers;
	layers.push_back(std::make_shared<NeuralInputLayer>(
	    input_width, input_height, input_num_channels));
	layers.push_back(hidden_layer);
	layers.push_back(decoding_layer);

	set_layers(layers);

	quick_connect();

	hidden_layer->autoencoder_position = NLAP_ENCODING;
	decoding_layer->autoencoder_position = NLAP_DECODING;

	initialize_neural_network(sigma);
}


bool Autoencoder::train(std::shared_ptr<Features> data)
{
	require(data != NULL, "Invalid (NULL) feature pointer");

	SGMatrix<float64_t> inputs = features_to_matrix(data);

	if (m_noise_type==AENT_DROPOUT)
		m_dropout_input = m_noise_parameter;
	if (m_noise_type==AENT_GAUSSIAN)
	{
		auto input_layer = std::static_pointer_cast<NeuralInputLayer>(get_layer(0));
		input_layer->gaussian_noise = m_noise_parameter;
	}

	for (int32_t i=0; i<m_num_layers-1; i++)
	{
		get_layer(i)->dropout_prop =
			get_layer(i)->is_input() ? m_dropout_input : m_dropout_hidden;
	}
	get_layer(m_num_layers-1)->dropout_prop = 0.0;

	m_is_training = true;
	for (int32_t i=0; i<m_num_layers; i++)
		get_layer(i)->is_training = true;

	bool result = false;
	if (m_optimization_method==NNOM_GRADIENT_DESCENT)
		result = train_gradient_descent(inputs, inputs);
	else if (m_optimization_method==NNOM_LBFGS)
		result = train_lbfgs(inputs, inputs);

	for (int32_t i=0; i<m_num_layers; i++)
		get_layer(i)->is_training = false;
	m_is_training = false;

	if (m_noise_type==AENT_GAUSSIAN)
	{
		auto input_layer = std::static_pointer_cast<NeuralInputLayer>(get_layer(0));
		input_layer->gaussian_noise = 0;
	}

	return result;
}

std::shared_ptr<DenseFeatures< float64_t >> Autoencoder::transform(
	std::shared_ptr<DenseFeatures< float64_t >> data)
{
	SGMatrix<float64_t> hidden_activation = forward_propagate(data, m_num_layers-2);
	return std::make_shared<DenseFeatures<float64_t>>(hidden_activation);
}

std::shared_ptr<DenseFeatures< float64_t >> Autoencoder::reconstruct(
	std::shared_ptr<DenseFeatures< float64_t >> data)
{
	SGMatrix<float64_t> reconstructed = forward_propagate(data);
	return std::make_shared<DenseFeatures<float64_t>>(reconstructed);
}

float64_t Autoencoder::compute_error(SGMatrix< float64_t > targets)
{
	float64_t error = NeuralNetwork::compute_error(targets);

	if (m_contraction_coefficient != 0.0)
		error +=
			get_layer(1)->compute_contraction_term(get_section(m_params,1));

	return error;
}

template <class T>
SGVector<T> Autoencoder::get_section(SGVector<T> v, int32_t i)
{
	return SGVector<T>(v.vector+m_index_offsets[i],
		get_layer(i)->get_num_parameters(), false);
}

void Autoencoder::init()
{
	m_noise_type = AENT_NONE;
	m_noise_parameter = 0.0;
	m_contraction_coefficient = 0.0;

	SG_ADD(&m_noise_parameter, "noise_parameter", "Noise Parameter");
	SG_ADD(
	    &m_contraction_coefficient, "contraction_coefficient",
	    "Contraction Coefficient");
	SG_ADD_OPTIONS(
	    (machine_int_t*)&m_noise_type, "noise_type", "Noise Type",
	    ParameterProperties::NONE,
	    SG_OPTIONS(AENT_NONE, AENT_DROPOUT, AENT_GAUSSIAN));
}
