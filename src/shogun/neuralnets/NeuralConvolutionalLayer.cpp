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

#include <shogun/neuralnets/NeuralConvolutionalLayer.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGVector.h>

using namespace shogun;

CNeuralConvolutionalLayer::CNeuralConvolutionalLayer() : CNeuralLayer()
{
	init();
}

CNeuralConvolutionalLayer::CNeuralConvolutionalLayer(
		EConvMapActivationFunction function,
		int32_t num_maps,
		int32_t input_width, int32_t input_height,
		int32_t radius_x, int32_t radius_y,
		int32_t pooling_width, int32_t pooling_height,
		int32_t stride_x, int32_t stride_y) : CNeuralLayer()
{
	init();
	m_num_maps = num_maps;
	m_input_width = input_width;
	m_input_height = input_height;
	m_radius_x = radius_x;
	m_radius_y = radius_y;
	m_pooling_width = pooling_width;
	m_pooling_height = pooling_height;
	m_stride_x = stride_x;
	m_stride_y = stride_y;
}

void CNeuralConvolutionalLayer::set_batch_size(int32_t batch_size)
{
	CNeuralLayer::set_batch_size(batch_size);
	
	if (autoencoder_position==NLAP_NONE)	
		m_convolution_output = SGMatrix<float64_t>(m_num_maps*
			(m_input_width/m_stride_x)*(m_input_height/m_stride_y), batch_size);
	else
		m_convolution_output = SGMatrix<float64_t>(
			m_num_maps*m_input_width*m_input_height, batch_size);
	
	m_max_indices = SGMatrix<float64_t>(m_num_neurons, m_batch_size);
	
	m_convolution_output_gradients = SGMatrix<float64_t>(
		m_convolution_output.num_rows, m_convolution_output.num_cols);
}


void CNeuralConvolutionalLayer::initialize(CDynamicObjectArray* layers, 
		SGVector< int32_t > input_indices)
{
	if (autoencoder_position==NLAP_NONE)
		m_num_neurons = (m_input_width/(m_stride_x*m_pooling_width)) * 
			(m_input_height/(m_stride_y*m_pooling_height)) * m_num_maps;
	else
		m_num_neurons = m_input_width*m_input_height*m_num_maps;
	
	CNeuralLayer::initialize(layers, input_indices);
	
	m_input_num_channels = 0;
	for (int32_t l=0; l<input_indices.vlen; l++)
	{
		CNeuralLayer* layer = 
			(CNeuralLayer*)layers->element(input_indices[l]);
		
		m_input_num_channels += layer->get_num_neurons()/(m_input_height*m_input_width);
		
		SG_UNREF(layer);
	}
	
	// one bias for each map and one weight matrix between each map in this 
	// layer and each channel in the input layers
	m_num_parameters = 
		m_num_maps*(1 + m_input_num_channels*(2*m_radius_x+1)*(2*m_radius_y+1));
}

void CNeuralConvolutionalLayer::initialize_parameters(SGVector<float64_t> parameters,
		SGVector<bool> parameter_regularizable,
		float64_t sigma)
{
	int32_t num_parameters_per_map = 
		1 + m_input_num_channels*(2*m_radius_x+1)*(2*m_radius_y+1);

	for (int32_t m=0; m<m_num_maps; m++)
	{
		float64_t* map_params = parameters.vector+m*num_parameters_per_map;
		bool* map_param_regularizable = 
			parameter_regularizable.vector+m*num_parameters_per_map;

		for (int32_t i=0; i<num_parameters_per_map; i++)
		{
			map_params[i] = CMath::normal_random(0.0, sigma);

			// turn off regularization for the bias, on for the rest of the parameters
			map_param_regularizable[i] = (i != 0);
		}
	}
}

void CNeuralConvolutionalLayer::compute_activations(
		SGVector<float64_t> parameters,
		CDynamicObjectArray* layers)
{
	int32_t num_parameters_per_map = 
		1 + m_input_num_channels*(2*m_radius_x+1)*(2*m_radius_y+1);
	
	for (int32_t m=0; m<m_num_maps; m++)
	{
		SGVector<float64_t> map_params(
			parameters.vector+m*num_parameters_per_map, 
			num_parameters_per_map, false);
		
		CConvolutionalFeatureMap map(m_input_width, m_input_height, 
			m_radius_x, m_radius_y, m_stride_x, m_stride_y, m, 
			m_activation_function, autoencoder_position);
		
		map.compute_activations(map_params, layers, m_input_indices, 
			m_convolution_output);
		
		map.pool_activations(m_convolution_output, 
			m_pooling_width, m_pooling_height, m_activations, m_max_indices);
	}
}

void CNeuralConvolutionalLayer::compute_gradients(
		SGVector<float64_t> parameters, 
		SGMatrix<float64_t> targets,
		CDynamicObjectArray* layers,
		SGVector<float64_t> parameter_gradients)
{
	if (targets.num_rows != 0)
	{
		// sqaured error measure
		// local_gradients = activations-targets
		int32_t length = m_num_neurons*m_batch_size;
		for (int32_t i=0; i<length; i++)
			m_activation_gradients[i] = (m_activations[i]-targets[i])/m_batch_size;
	}
	
	if (dropout_prop>0.0)
	{
		int32_t len = m_num_neurons*m_batch_size;
		for (int32_t i=0; i<len; i++)
			m_activation_gradients[i] *= m_dropout_mask[i];
	}
	
	// compute the pre-pooling activation gradients
	m_convolution_output_gradients.zero();
	for (int32_t i=0; i<m_num_neurons; i++)
		for (int32_t j=0; j<m_batch_size; j++)
			if (m_max_indices(i,j)!=-1.0)
				m_convolution_output_gradients(m_max_indices(i,j),j) = 
					m_activation_gradients(i,j);

	int32_t num_parameters_per_map =
		1 + m_input_num_channels*(2*m_radius_x+1)*(2*m_radius_y+1);
	
	for (int32_t m=0; m<m_num_maps; m++)
	{
		SGVector<float64_t> map_params(
			parameters.vector+m*num_parameters_per_map, 
			num_parameters_per_map, false);
		
		SGVector<float64_t> map_gradients(
			parameter_gradients.vector+m*num_parameters_per_map, 
			num_parameters_per_map, false);
		
		CConvolutionalFeatureMap map(m_input_width, m_input_height, 
			m_radius_x, m_radius_y, m_stride_x, m_stride_y, m, 
			m_activation_function, autoencoder_position);
			
		map.compute_gradients(map_params, m_convolution_output, 
			m_convolution_output_gradients, layers, 
			m_input_indices, map_gradients);
	}
}

float64_t CNeuralConvolutionalLayer::compute_error(SGMatrix<float64_t> targets)
{
	// error = 0.5*(sum(targets-activations)^2)/batch_size
	float64_t sum = 0;
	int32_t length = m_num_neurons*m_batch_size;
	for (int32_t i=0; i<length; i++)
		sum += (targets[i]-m_activations[i])*(targets[i]-m_activations[i]);
	sum *= (0.5/m_batch_size);
	return sum;
}

void CNeuralConvolutionalLayer::enforce_max_norm(SGVector<float64_t> parameters, 
		float64_t max_norm)
{
	int32_t num_weights = (2*m_radius_x+1)*(2*m_radius_y+1);	
	
	int32_t num_parameters_per_map = 1 + m_input_num_channels*num_weights;
	
	for (int32_t offset=1; offset<parameters.vlen; offset+=num_parameters_per_map)
	{
		float64_t* weights = parameters.vector+offset;
		
		float64_t norm = 
				SGVector<float64_t>::twonorm(weights, num_weights);
			
		if (norm > max_norm)
		{
			float64_t multiplier = max_norm/norm;
			for (int32_t i=0; i<num_weights; i++)
				weights[i] *= multiplier;
		}
	}
}

void CNeuralConvolutionalLayer::init()
{
	m_num_maps = 1;
	m_input_width = 0;
	m_input_height = 0;
	m_input_num_channels = 0;
	m_radius_x = 0;
	m_radius_y = 0;
	m_pooling_width = 1;
	m_pooling_height = 1;
	m_stride_x = 1;
	m_stride_y = 1;
	m_activation_function = CMAF_IDENTITY;
	
	SG_ADD(&m_num_maps, "num_maps", "Number of maps", MS_NOT_AVAILABLE);
	SG_ADD(&m_input_width, "input_width", "Input Width", MS_NOT_AVAILABLE);
	SG_ADD(&m_input_height, "input_height", "Input Height", MS_NOT_AVAILABLE);
	SG_ADD(&m_input_num_channels, "input_num_channels", "Input's number of channels", 
		MS_NOT_AVAILABLE);
	SG_ADD(&m_radius_x, "radius_x", "X Radius", MS_NOT_AVAILABLE);
	SG_ADD(&m_radius_y, "radius_y", "Y Radius", MS_NOT_AVAILABLE);
	SG_ADD(&m_pooling_width, "pooling_width", "Pooling Width", MS_NOT_AVAILABLE);
	SG_ADD(&m_pooling_height, "pooling_height", "Pooling Height", MS_NOT_AVAILABLE);
	SG_ADD(&m_stride_x, "stride_x", "X Stride", MS_NOT_AVAILABLE);
	SG_ADD(&m_stride_y, "stride_y", "Y Stride", MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*) &m_activation_function, "activation_function", 
		"Activation Function", MS_NOT_AVAILABLE);
	
	SG_ADD(&m_convolution_output, "convolution_output", 
		"Convolution Output", MS_NOT_AVAILABLE);
	
	SG_ADD(&m_convolution_output_gradients, "convolution_output_gradients", 
		"Convolution Output Gradients", MS_NOT_AVAILABLE);
}
