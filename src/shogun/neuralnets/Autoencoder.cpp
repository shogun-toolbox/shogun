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
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

CAutoencoder::CAutoencoder() : CNeuralNetwork()
{
	init();
}

CAutoencoder::CAutoencoder(int32_t num_inputs, CNeuralLayer* hidden_layer,
	CNeuralLayer* decoding_layer, float64_t sigma) : CNeuralNetwork()
{
	init();

	if (decoding_layer==NULL)
		decoding_layer = new CNeuralLinearLayer(num_inputs);

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(new CNeuralInputLayer(num_inputs));
	layers->append_element(hidden_layer);
	layers->append_element(decoding_layer);

	set_layers(layers);

	quick_connect();

	hidden_layer->autoencoder_position = NLAP_ENCODING;
	decoding_layer->autoencoder_position = NLAP_DECODING;

	initialize_neural_network(sigma);
}

CAutoencoder::CAutoencoder(
	int32_t input_width, int32_t input_height, int32_t input_num_channels,
	CNeuralConvolutionalLayer* hidden_layer,
	CNeuralConvolutionalLayer* decoding_layer,
	float64_t sigma)
	: CNeuralNetwork()
{
	init();

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(new CNeuralInputLayer(input_width, input_height, input_num_channels));
	layers->append_element(hidden_layer);
	layers->append_element(decoding_layer);

	set_layers(layers);

	quick_connect();

	hidden_layer->autoencoder_position = NLAP_ENCODING;
	decoding_layer->autoencoder_position = NLAP_DECODING;

	initialize_neural_network(sigma);
}


bool CAutoencoder::train(CFeatures* data)
{
	REQUIRE(data != NULL, "Invalid (NULL) feature pointer\n");

	SGMatrix<float64_t> inputs = features_to_matrix(data);

	if (m_noise_type==AENT_DROPOUT)
		m_dropout_input = m_noise_parameter;
	if (m_noise_type==AENT_GAUSSIAN)
	{
		CNeuralInputLayer* input_layer = (CNeuralInputLayer*)get_layer(0);
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
		CNeuralInputLayer* input_layer = (CNeuralInputLayer*)get_layer(0);
		input_layer->gaussian_noise = 0;
	}

	return result;
}

CDenseFeatures< float64_t >* CAutoencoder::transform(
	CDenseFeatures< float64_t >* data)
{
	SGMatrix<float64_t> hidden_activation = forward_propagate(data, m_num_layers-2);
	return new CDenseFeatures<float64_t>(hidden_activation);
}

CDenseFeatures< float64_t >* CAutoencoder::reconstruct(
	CDenseFeatures< float64_t >* data)
{
	SGMatrix<float64_t> reconstructed = forward_propagate(data);
	return new CDenseFeatures<float64_t>(reconstructed);
}

float64_t CAutoencoder::compute_error(SGMatrix< float64_t > targets)
{
	float64_t error = CNeuralNetwork::compute_error(targets);

	if (m_contraction_coefficient != 0.0)
		error +=
			get_layer(1)->compute_contraction_term(get_section(m_params,1));

	return error;
}

template <class T>
SGVector<T> CAutoencoder::get_section(SGVector<T> v, int32_t i)
{
	return SGVector<T>(v.vector+m_index_offsets[i],
		get_layer(i)->get_num_parameters(), false);
}

void CAutoencoder::init()
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
