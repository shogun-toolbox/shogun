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
#include <shogun/lib/DynamicObjectArray.h>
#include <features/DenseFeatures.h>

using namespace shogun;

CDeepAutoencoder::CDeepAutoencoder() : CAutoencoder()
{
	init();
}

CDeepAutoencoder::CDeepAutoencoder(CDynamicObjectArray* layers, float64_t sigma): 
CAutoencoder()
{
	set_layers(layers);
	init();
	m_sigma = sigma;
	quick_connect();
	initialize(m_sigma);
}

void CDeepAutoencoder::set_layers(CDynamicObjectArray* layers)
{
	CNeuralNetwork::set_layers(layers);
	
	for (int32_t i=0; i<m_num_layers; i++)
	{
		REQUIRE(get_layer(i)->get_num_neurons()==get_layer(m_num_layers-i-1)->get_num_neurons(),
			"Layer %i (%i neurons) must have the same number of neurons "
			"as layer %i (%i neurons)\n", i, get_layer(i)->get_num_neurons(),
			m_num_layers-i-1, get_layer(m_num_layers-i-1)->get_num_neurons());
	}
}

void CDeepAutoencoder::pre_train(CFeatures* data)
{
	SGMatrix<float64_t> data_matrix = features_to_matrix(data);
	
	int32_t num_encoding_layers = (m_num_layers-1)/2;
	for (int32_t i=1; i<=num_encoding_layers; i++)
	{
		SG_INFO("Pre-training Layer %i\n", i);
		
		CAutoencoder ae(get_layer(i-1)->get_num_neurons(), 
			get_layer(i), get_layer(m_num_layers-i), m_sigma);
		
		ae.noise_type = EAENoiseType(pt_noise_type[i-1]);
		ae.noise_parameter = pt_noise_parameter[i-1];
		ae.set_contraction_coefficient(pt_contraction_coefficient[i-1]);
		ae.optimization_method = ENNOptimizationMethod(pt_optimization_method[i-1]);
		ae.l2_coefficient = pt_l2_coefficient[i-1];
		ae.l1_coefficient = pt_l1_coefficient[i-1];
		ae.epsilon = pt_epsilon[i-1];
		ae.max_num_epochs = pt_max_num_epochs[i-1];
		ae.gd_learning_rate = pt_gd_learning_rate[i-1];
		ae.gd_learning_rate_decay = pt_gd_learning_rate_decay[i-1];
		ae.gd_momentum = pt_gd_momentum[i-1];
		ae.gd_mini_batch_size = pt_gd_mini_batch_size[i-1];
		ae.gd_error_damping_coeff = pt_gd_error_damping_coeff[i-1];
		
		// forward propagate the data to obtain the training data for the 
		// current autoencoder
		for (int32_t j=0; j<i; j++)
			get_layer(j)->set_batch_size(data_matrix.num_cols);
		SGMatrix<float64_t> ae_input_matrix = forward_propagate(data_matrix, i-1);
		CDenseFeatures<float64_t> ae_input_features(ae_input_matrix);
		for (int32_t j=0; j<i-1; j++)
			get_layer(j)->set_batch_size(1);
		
		ae.train(&ae_input_features);
		
		SGVector<float64_t> ae_params = ae.get_parameters();
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
}

CDenseFeatures< float64_t >* CDeepAutoencoder::transform(
	CDenseFeatures< float64_t >* data)
{
	SGMatrix<float64_t> transformed = forward_propagate(data, (m_num_layers-1)/2);
	return new CDenseFeatures<float64_t>(transformed); 
}

CDenseFeatures< float64_t >* CDeepAutoencoder::reconstruct(
	CDenseFeatures< float64_t >* data)
{
	SGMatrix<float64_t> reconstructed = forward_propagate(data);
	return new CDenseFeatures<float64_t>(reconstructed); 
}

CNeuralNetwork* CDeepAutoencoder::convert_to_neural_network(
	CNeuralLayer* output_layer, float64_t sigma)
{
	CDynamicObjectArray* layers = new CDynamicObjectArray;
	for (int32_t i=0; i<=(m_num_layers-1)/2; i++)
		layers->append_element(get_layer(i));
	
	if (output_layer != NULL)
		layers->append_element(output_layer);
	
	CNeuralNetwork* net = new CNeuralNetwork(layers);
	net->quick_connect();
	net->initialize(sigma);
	
	SGVector<float64_t> net_params = net->get_parameters();
	
	int32_t len = m_index_offsets[(m_num_layers-1)/2]
		+ get_layer((m_num_layers-1)/2)->get_num_parameters();
		
	for (int32_t i=0; i<len; i++)
		net_params[i] = m_params[i];
	
	return net;
}

float64_t CDeepAutoencoder::compute_error(SGMatrix< float64_t > targets)
{
	float64_t error = CNeuralNetwork::compute_error(targets);
	
	if (m_contraction_coefficient != 0.0)
	
	for (int32_t i=1; i<=(m_num_layers-1)/2; i++)
		error += 
			get_layer(i)->compute_contraction_term(get_section(m_params,i)); 
		
	return error;
}

void CDeepAutoencoder::set_contraction_coefficient(float64_t coeff)
{
	m_contraction_coefficient = coeff;
	for (int32_t i=1; i<=(m_num_layers-1)/2; i++)
		get_layer(i)->contraction_coefficient = coeff;
}


template <class T>
SGVector<T> CDeepAutoencoder::get_section(SGVector<T> v, int32_t i)
{
	return SGVector<T>(v.vector+m_index_offsets[i], 
		get_layer(i)->get_num_parameters(), false);
}

void CDeepAutoencoder::init()
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
		"Pre-training Noise Type", MS_NOT_AVAILABLE);
	SG_ADD(&pt_noise_parameter, "pt_noise_parameter", 
		"Pre-training Noise Parameter", MS_NOT_AVAILABLE);
	SG_ADD(&pt_contraction_coefficient, "pt_contraction_coefficient",
	    "Pre-training Contraction Coefficient", MS_NOT_AVAILABLE);
	SG_ADD(&pt_optimization_method, "pt_optimization_method",
	    "Pre-training Optimization Method", MS_NOT_AVAILABLE);
	SG_ADD(&pt_gd_mini_batch_size, "pt_gd_mini_batch_size",
	    "Pre-training Gradient Descent Mini-batch size", MS_NOT_AVAILABLE);
	SG_ADD(&pt_max_num_epochs, "pt_max_num_epochs",
	    "Pre-training Max number of Epochs", MS_NOT_AVAILABLE);
	SG_ADD(&pt_gd_learning_rate, "pt_gd_learning_rate",
	    "Pre-training Gradient descent learning rate", MS_NOT_AVAILABLE);
	SG_ADD(&pt_gd_learning_rate_decay, "pt_gd_learning_rate_decay",
	    "Pre-training Gradient descent learning rate decay", MS_NOT_AVAILABLE);
	SG_ADD(&pt_gd_momentum, "pt_gd_momentum",
	    "Pre-training Gradient Descent Momentum", MS_NOT_AVAILABLE);
	SG_ADD(&pt_gd_error_damping_coeff, "pt_gd_error_damping_coeff",
	    "Pre-training Gradient Descent Error Damping Coeff", MS_NOT_AVAILABLE);
	SG_ADD(&pt_epsilon, "pt_epsilon",
	    "Pre-training Epsilon", MS_NOT_AVAILABLE);
	SG_ADD(&pt_l2_coefficient, "pt_l2_coefficient",
	    "Pre-training L2 regularization coeff", MS_NOT_AVAILABLE);
	SG_ADD(&pt_l1_coefficient, "pt_l1_coefficient",
	    "Pre-training L1 regularization coeff", MS_NOT_AVAILABLE);
	
	SG_ADD(&m_sigma, "m_sigma", "Initialization Sigma", MS_NOT_AVAILABLE);
}
