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

#include <shogun/neuralnets/DeepAutoencoder.h>
#include <shogun/features/DenseFeatures.h>

#include <shogun/neuralnets/NeuralLinearLayer.h>
#include <shogun/neuralnets/NeuralLogisticLayer.h>
#include <shogun/neuralnets/NeuralRectifiedLinearLayer.h>
#include <shogun/neuralnets/NeuralConvolutionalLayer.h>

#include <string>

using namespace shogun;

DeepAutoencoder::DeepAutoencoder() : Autoencoder()
{
	init();
}

DeepAutoencoder::DeepAutoencoder(
    const std::vector<std::shared_ptr<NeuralLayer>>& layers, float64_t sigma)
    : Autoencoder()
{
	set_layers(layers);
	init();
	m_sigma = sigma;
	quick_connect();

	int32_t num_encoding_layers = (m_num_layers-1)/2;
	for (int32_t i=0; i<m_num_layers; i++)
	{
		if (i<= num_encoding_layers)
			get_layer(i)->autoencoder_position = NLAP_ENCODING;
		else
			get_layer(i)->autoencoder_position = NLAP_DECODING;
	}

	initialize_neural_network(m_sigma);

	for (int32_t i=0; i<m_num_layers; i++)
	{
		require(get_layer(i)->get_num_neurons()==get_layer(m_num_layers-i-1)->get_num_neurons(),
			"Layer {} ({} neurons) must have the same number of neurons "
			"as layer {} ({} neurons)", i, get_layer(i)->get_num_neurons(),
			m_num_layers-i-1, get_layer(m_num_layers-i-1)->get_num_neurons());
	}
}

void DeepAutoencoder::pre_train(std::shared_ptr<Features> data)
{
	SGMatrix<float64_t> data_matrix = features_to_matrix(data);

	int32_t num_encoding_layers = (m_num_layers-1)/2;
	for (int32_t i=1; i<=num_encoding_layers; i++)
	{
		io::info("Pre-training Layer {}", i);

		auto ae_encoding_layer = get_layer(i)->clone()->as<NeuralLayer>();

		auto ae_decoding_layer =
			get_layer(m_num_layers-i)->clone()->as<NeuralLayer>();

		std::shared_ptr<Autoencoder> ae = NULL;

		if (strcmp(ae_encoding_layer->get_name(), "NeuralConvolutionalLayer")==0)
		{
			ae = std::make_shared<Autoencoder>(
				ae_encoding_layer->get_width(),
				ae_encoding_layer->get_height(),
				get_layer(i-1)->get_num_neurons()
				/(ae_encoding_layer->get_width()*ae_encoding_layer->get_height()),
				ae_encoding_layer->as<NeuralConvolutionalLayer>(),
				ae_decoding_layer->as<NeuralConvolutionalLayer>(), m_sigma);
		}
		else
		{
			ae = std::make_shared<Autoencoder>(get_layer(i-1)->get_num_neurons(),
				ae_encoding_layer, ae_decoding_layer, m_sigma);
		}




		ae->set_noise_type(EAENoiseType(pt_noise_type[i-1]));
		ae->set_noise_parameter(pt_noise_parameter[i-1]);
		ae->set_contraction_coefficient(pt_contraction_coefficient[i-1]);
		ae->set_optimization_method(ENNOptimizationMethod(pt_optimization_method[i-1]));
		ae->set_l2_coefficient(pt_l2_coefficient[i-1]);
		ae->set_l1_coefficient(pt_l1_coefficient[i-1]);
		ae->set_epsilon(pt_epsilon[i-1]);
		ae->set_max_num_epochs(pt_max_num_epochs[i-1]);
		ae->set_gd_learning_rate(pt_gd_learning_rate[i-1]);
		ae->set_gd_learning_rate_decay(pt_gd_learning_rate_decay[i-1]);
		ae->set_gd_momentum(pt_gd_momentum[i-1]);
		ae->set_gd_mini_batch_size(pt_gd_mini_batch_size[i-1]);
		ae->set_gd_error_damping_coeff(pt_gd_error_damping_coeff[i-1]);

		// forward propagate the data to obtain the training data for the
		// current autoencoder
		for (int32_t j=0; j<i; j++)
			get_layer(j)->set_batch_size(data_matrix.num_cols);
		SGMatrix<float64_t> ae_input_matrix = forward_propagate(data_matrix, i-1);
		auto ae_input_features = std::make_shared<DenseFeatures<float64_t>>(ae_input_matrix);
		for (int32_t j=0; j<i-1; j++)
			get_layer(j)->set_batch_size(1);

		ae->train(ae_input_features);

		SGVector<float64_t> ae_params = ae->get_parameters();
		SGVector<float64_t> encoding_layer_params = get_section(m_params, i);
		SGVector<float64_t> decoding_layer_params = get_section(m_params, m_num_layers-i);

		for (int32_t j=0; j<ae_params.vlen;j++)
		{
			if (j<encoding_layer_params.vlen)
				encoding_layer_params[j] = ae_params[j];
			else
				decoding_layer_params[j-encoding_layer_params.vlen] = ae_params[j];
		}

	}

	set_batch_size(1);
}

std::shared_ptr<DenseFeatures< float64_t >> DeepAutoencoder::transform(
	std::shared_ptr<DenseFeatures< float64_t >> data)
{
	SGMatrix<float64_t> transformed = forward_propagate(data, (m_num_layers-1)/2);
	return std::make_shared<DenseFeatures<float64_t>>(transformed);
}

std::shared_ptr<DenseFeatures< float64_t >> DeepAutoencoder::reconstruct(
	std::shared_ptr<DenseFeatures< float64_t >> data)
{
	SGMatrix<float64_t> reconstructed = forward_propagate(data);
	return std::make_shared<DenseFeatures<float64_t>>(reconstructed);
}

std::shared_ptr<NeuralNetwork> DeepAutoencoder::convert_to_neural_network(
	std::shared_ptr<NeuralLayer> output_layer, float64_t sigma)
{
	std::vector<std::shared_ptr<NeuralLayer>> layers;
	for (int32_t i=0; i<=(m_num_layers-1)/2; i++)
	{
		auto layer = get_layer(i)->clone()->as<NeuralLayer>();
		layer->autoencoder_position = NLAP_NONE;
		layers.push_back(layer);
	}

	if (output_layer != NULL)
		layers.push_back(output_layer);

	auto net = std::make_shared<NeuralNetwork>(layers);
	net->quick_connect();
	net->initialize_neural_network(sigma);

	SGVector<float64_t> net_params = net->get_parameters();

	int32_t len = m_index_offsets[(m_num_layers-1)/2]
		+ get_layer((m_num_layers-1)/2)->get_num_parameters();

	for (int32_t i=0; i<len; i++)
		net_params[i] = m_params[i];

	return net;
}

float64_t DeepAutoencoder::compute_error(SGMatrix< float64_t > targets)
{
	float64_t error = NeuralNetwork::compute_error(targets);

	if (m_contraction_coefficient != 0.0)

	for (int32_t i=1; i<=(m_num_layers-1)/2; i++)
		error +=
			get_layer(i)->compute_contraction_term(get_section(m_params,i));

	return error;
}

void DeepAutoencoder::set_contraction_coefficient(float64_t coeff)
{
	m_contraction_coefficient = coeff;
	for (int32_t i=1; i<=(m_num_layers-1)/2; i++)
		get_layer(i)->contraction_coefficient = coeff;
}


template <class T>
SGVector<T> DeepAutoencoder::get_section(SGVector<T> v, int32_t i)
{
	return SGVector<T>(v.vector+m_index_offsets[i],
		get_layer(i)->get_num_parameters(), false);
}

void DeepAutoencoder::init()
{
	m_sigma = 0.01;

	pt_noise_type = SGVector<int32_t>((m_num_layers-1)/2);
	pt_noise_type.set_const(AENT_NONE);

	pt_noise_parameter = SGVector<float64_t>((m_num_layers-1)/2);
	pt_noise_parameter.set_const(0.0);

	pt_contraction_coefficient = SGVector<float64_t>((m_num_layers-1)/2);
	pt_contraction_coefficient.set_const(0.0);

	pt_optimization_method = SGVector<int32_t>((m_num_layers-1)/2);
	pt_optimization_method.set_const(NNOM_LBFGS);

	pt_l2_coefficient = SGVector<float64_t>((m_num_layers-1)/2);
	pt_l2_coefficient.set_const(0.0);

	pt_l1_coefficient = SGVector<float64_t>((m_num_layers-1)/2);
	pt_l1_coefficient.set_const(0.0);

	pt_epsilon = SGVector<float64_t>((m_num_layers-1)/2);
	pt_epsilon.set_const(1e-5);

	pt_max_num_epochs = SGVector<int32_t>((m_num_layers-1)/2);
	pt_max_num_epochs.set_const(0);

	pt_gd_mini_batch_size = SGVector<int32_t>((m_num_layers-1)/2);
	pt_gd_mini_batch_size.set_const(0);

	pt_gd_learning_rate = SGVector<float64_t>((m_num_layers-1)/2);
	pt_gd_learning_rate.set_const(0.1);

	pt_gd_learning_rate_decay = SGVector<float64_t>((m_num_layers-1)/2);
	pt_gd_learning_rate_decay.set_const(1.0);

	pt_gd_momentum = SGVector<float64_t>((m_num_layers-1)/2);
	pt_gd_momentum.set_const(0.9);

	pt_gd_error_damping_coeff = SGVector<float64_t>((m_num_layers-1)/2);
	pt_gd_error_damping_coeff.set_const(-1);

	SG_ADD(&pt_noise_type, "pt_noise_type",
		"Pre-training Noise Type");
	SG_ADD(&pt_noise_parameter, "pt_noise_parameter",
		"Pre-training Noise Parameter");
	SG_ADD(&pt_contraction_coefficient, "pt_contraction_coefficient",
	    "Pre-training Contraction Coefficient");
	SG_ADD(&pt_optimization_method, "pt_optimization_method",
	    "Pre-training Optimization Method");
	SG_ADD(&pt_gd_mini_batch_size, "pt_gd_mini_batch_size",
	    "Pre-training Gradient Descent Mini-batch size");
	SG_ADD(&pt_max_num_epochs, "pt_max_num_epochs",
	    "Pre-training Max number of Epochs");
	SG_ADD(&pt_gd_learning_rate, "pt_gd_learning_rate",
	    "Pre-training Gradient descent learning rate");
	SG_ADD(&pt_gd_learning_rate_decay, "pt_gd_learning_rate_decay",
	    "Pre-training Gradient descent learning rate decay");
	SG_ADD(&pt_gd_momentum, "pt_gd_momentum",
	    "Pre-training Gradient Descent Momentum");
	SG_ADD(&pt_gd_error_damping_coeff, "pt_gd_error_damping_coeff",
	    "Pre-training Gradient Descent Error Damping Coeff");
	SG_ADD(&pt_epsilon, "pt_epsilon",
	    "Pre-training Epsilon");
	SG_ADD(&pt_l2_coefficient, "pt_l2_coefficient",
	    "Pre-training L2 regularization coeff");
	SG_ADD(&pt_l1_coefficient, "pt_l1_coefficient",
	    "Pre-training L1 regularization coeff");

	SG_ADD(&m_sigma, "m_sigma", "Initialization Sigma");
}
