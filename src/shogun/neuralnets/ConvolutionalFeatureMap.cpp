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

#include <shogun/neuralnets/ConvolutionalFeatureMap.h>
#include <shogun/neuralnets/NeuralLayer.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CConvolutionalFeatureMap::CConvolutionalFeatureMap(
	int32_t input_width, int32_t input_height, 
	int32_t radius_x, int32_t radius_y,
	int32_t stride_x, int32_t stride_y,
	int32_t index, 
	EConvMapActivationFunction function) :
		m_input_width(input_width), m_input_height(input_height), 
		m_radius_x(radius_x), m_radius_y(radius_y),
		m_stride_x(stride_x), m_stride_y(stride_y),
		m_index(index),
		m_activation_function(function)
{
	m_output_width = m_input_width/m_stride_x;
	m_output_height = m_input_height/m_stride_y;
	
	m_input_num_neurons = m_input_width*m_input_height;
	m_output_num_neurons = m_output_width*m_output_height;
	
	m_row_offset = m_index*m_output_num_neurons;
	
	m_filter_width = 2*m_radius_x+1;
	m_filter_height = 2*m_radius_y+1;
}

void CConvolutionalFeatureMap::compute_activations(
	SGVector< float64_t > parameters, 
	CDynamicObjectArray* layers, 
	SGVector< int32_t > input_indices,
	SGMatrix<float64_t> activations)
{
	int32_t batch_size = activations.num_cols;
	
	float64_t bias = parameters[0];
	for (int32_t i=0; i<m_output_num_neurons; i++)
	{
		for (int32_t j=0; j<batch_size; j++)
		{
			activations(i+m_row_offset,j) = bias;
		}
	}
	
	int32_t weights_index_offset = 1;
	for (int32_t l=0; l<input_indices.vlen; l++)
	{
		CNeuralLayer* layer = 
			(CNeuralLayer*)layers->element(input_indices[l]);
		
		int32_t num_maps = layer->get_num_neurons()/m_input_num_neurons;
		
		for (int32_t m=0; m<num_maps; m++)
		{
			SGMatrix<float64_t> weights_matrix(parameters.vector+weights_index_offset, 
				m_filter_height, m_filter_width, false);
			weights_index_offset += m_filter_height*m_filter_width;
			
			convolve(layer->get_activations(), weights_matrix, activations, 
				false, false, m*m_input_num_neurons, m_row_offset);
		}
		
		SG_UNREF(layer);
	}
	
	if (m_activation_function==CMAF_LOGISTIC)
	{
		for (int32_t i=0; i<m_output_num_neurons; i++)
			for (int32_t j=0; j<batch_size; j++)
				activations(i+m_row_offset,j) = 
					1.0/(1.0+CMath::exp(-1.0*activations(i+m_row_offset,j)));
	}
	else if (m_activation_function==CMAF_RECTIFIED_LINEAR)
	{
		for (int32_t i=0; i<m_output_num_neurons; i++)
			for (int32_t j=0; j<batch_size; j++)
				activations(i+m_row_offset,j) = 
					CMath::max<float64_t>(0, activations(i+m_row_offset,j));
	}
}

void CConvolutionalFeatureMap::compute_gradients(
	SGVector< float64_t > parameters,
	SGMatrix<float64_t> activations,
	SGMatrix< float64_t > activation_gradients, 
	CDynamicObjectArray* layers, 
	SGVector< int32_t > input_indices,
	SGVector< float64_t > parameter_gradients)
{
	int32_t batch_size = activation_gradients.num_cols;
	
	if (m_activation_function==CMAF_LOGISTIC)
	{
		for (int32_t i=0; i<m_output_num_neurons; i++)
		{
			for (int32_t j=0; j<batch_size; j++)
			{
				activation_gradients(i+m_row_offset,j) *= 
					activation_gradients(i+m_row_offset,j) * 
					(1.0-activation_gradients(i+m_row_offset,j));
			}
		}
	}
	else if (m_activation_function==CMAF_RECTIFIED_LINEAR)
	{
		for (int32_t i=0; i<m_output_num_neurons; i++)
			for (int32_t j=0; j<batch_size; j++)
				if (activations(i+m_row_offset,j)==0)
					activation_gradients(i+m_row_offset,j) = 0;
	}
	
	float64_t bias_gradient = 0;
	for (int32_t i=0; i<m_output_num_neurons; i++)
		for (int32_t j=0; j<batch_size; j++)
			bias_gradient += activation_gradients(i+m_row_offset,j);
	
	parameter_gradients[0] = bias_gradient;
	
	int32_t weights_index_offset = 1;
	for (int32_t l=0; l<input_indices.vlen; l++)
	{
		CNeuralLayer* layer = 
			(CNeuralLayer*)layers->element(input_indices[l]);
		
		int32_t num_maps = layer->get_num_neurons()/m_input_num_neurons;
		
		for (int32_t m=0; m<num_maps; m++)
		{
			SGMatrix<float64_t> W(parameters.vector+weights_index_offset, 
				m_filter_height, m_filter_width, false);
			SGMatrix<float64_t> WG(parameter_gradients.vector+weights_index_offset, 
				m_filter_height, m_filter_width, false);
			weights_index_offset += m_filter_height*m_filter_width;
			
			compute_weight_gradients(layer->get_activations(), 
				activation_gradients, WG, m*m_input_num_neurons, m_row_offset);
			
			if (!layer->is_input())
				convolve(activation_gradients, W, 
					layer->get_activation_gradients(), true, false, 
					m_row_offset, m*m_input_num_neurons);
		}	
		
		SG_UNREF(layer);
	}
}

void CConvolutionalFeatureMap::pool_activations(
	SGMatrix< float64_t > activations, 
	int32_t pooling_width, int32_t pooling_height, 
	SGMatrix< float64_t > pooled_activations, 
	SGMatrix< float64_t > max_indices)
{
	int32_t pooled_row_offset = m_row_offset/(pooling_width*pooling_height);
	
	for (int32_t i=0; i<pooled_activations.num_cols; i++)
	{
		SGMatrix<float64_t> image(
			activations.matrix+i*activations.num_rows + m_row_offset, 
			m_output_height, m_output_width, false);
		
		SGMatrix<float64_t> result(
			pooled_activations.matrix+i*pooled_activations.num_rows + pooled_row_offset, 
			m_output_height/pooling_height, m_output_width/pooling_width, false);
		
		SGMatrix<float64_t> indices(
			max_indices.matrix+i*max_indices.num_rows + pooled_row_offset, 
			m_output_height/pooling_height, m_output_width/pooling_width, false);
		
		for (int32_t x=0; x<m_output_width; x+=pooling_width)
		{
			for (int32_t y=0; y<m_output_height; y+=pooling_height)
			{
				float64_t max = image(y,x);
				int32_t max_index = m_row_offset+y+x*image.num_rows;
				
				for (int32_t x1=x; x1<x+pooling_width; x1++)
				{
					for (int32_t y1=y; y1<y+pooling_height; y1++)
					{
						if (image(y1,x1) > max)
						{
							max = image(y1,x1);
							max_index = m_row_offset+y1+x1*image.num_rows;
						}
					}
				}
				result(y/pooling_height, x/pooling_width) = max;
				indices(y/pooling_height, x/pooling_width) = max_index;
			}
		}
	}
}

void CConvolutionalFeatureMap::convolve(
	SGMatrix< float64_t > inputs, 
	SGMatrix< float64_t > weights, 
	SGMatrix< float64_t > outputs,
	bool flip,
	bool reset_output,
	int32_t inputs_row_offset,
 	int32_t outputs_row_offset)
{
	for (int32_t i=0; i<outputs.num_cols; i++)
	{
		SGMatrix<float64_t> image(
			inputs.matrix+i*inputs.num_rows + inputs_row_offset, 
			m_input_height, m_input_width, false);
		
		SGMatrix<float64_t> result(
			outputs.matrix+i*outputs.num_rows + outputs_row_offset, 
			m_output_height, m_output_width, false);
		
		for (int32_t x=0; x<m_input_width; x+=m_stride_x)
		{
			for (int32_t y=0; y<m_input_height; y+=m_stride_y)
			{
				float64_t sum = reset_output ? 0 : result(y/m_stride_y,x/m_stride_x);
				for (int32_t x1=x-m_radius_x; x1<=x+m_radius_x; x1++)
				{
					for (int32_t y1=y-m_radius_y; y1<=y+m_radius_y; y1++)
					{
						if (x1>=0 && y1>=0 && x1<image.num_cols && y1<image.num_rows)
						{
							if (flip)
								sum += 
									weights(y1-y+m_radius_y,x1-x+m_radius_x)*image(y1,x1);
							else
								sum += 
									weights(m_radius_y-y1+y,m_radius_x-x1+x)*image(y1,x1);
						}
					}
				}
				result(y/m_stride_y,x/m_stride_x) = sum;
			}
		}
	}
}

void CConvolutionalFeatureMap::compute_weight_gradients(
	SGMatrix< float64_t > inputs, 
	SGMatrix< float64_t > local_gradients, 
	SGMatrix< float64_t > weight_gradients,
	int32_t inputs_row_offset,
 	int32_t local_gradients_row_offset)
{
	weight_gradients.zero();
	for (int32_t i=0; i<local_gradients.num_cols; i++)
	{
		SGMatrix<float64_t> image(
			inputs.matrix+i*inputs.num_rows + inputs_row_offset, 
			m_input_height, m_input_width, false);
		
		SGMatrix<float64_t> LG_image(
			local_gradients.matrix+i*local_gradients.num_rows 
			+ local_gradients_row_offset, m_output_height, m_output_width, false);
		
		for (int32_t x=0; x<m_input_width; x+=m_stride_x)
		{
			for (int32_t y=0; y<m_input_height; y+=m_stride_y)
			{
				for (int32_t x1=x-m_radius_x; x1<=x+m_radius_x; x1++)
				{
					for (int32_t y1=y-m_radius_y; y1<=y+m_radius_y; y1++)
					{
						if (x1>=0 && y1>=0 && x1<image.num_cols && y1<image.num_rows)
							weight_gradients(m_radius_y-y1+y,m_radius_x-x1+x) +=
								LG_image(y/m_stride_y,x/m_stride_x)*image(y1,x1);
					}
				}
			}
		}
	}
}
