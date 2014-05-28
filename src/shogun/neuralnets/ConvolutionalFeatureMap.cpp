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
	int32_t width, int32_t height, 
	int32_t radius_x, int32_t radius_y, int32_t index) :
	m_width(width), m_height(height), 
	m_radius_x(radius_x), m_radius_y(radius_y), m_index(index)
{

}

void CConvolutionalFeatureMap::compute_activations(
	SGVector< float64_t > parameters, 
	CDynamicObjectArray* layers, 
	SGVector< int32_t > input_indices,
	SGMatrix<float64_t> activations,
	SGMatrix<float64_t> buffer)
{
	int32_t num_neurons = m_width*m_height;
	int32_t row_offset = m_index*num_neurons;
	int32_t batch_size = activations.num_cols;
	
	// sum up all the inputs into the buffer
	buffer.zero();	
	for (int32_t l=0; l<input_indices.vlen; l++)
	{
		CNeuralLayer* layer = 
			(CNeuralLayer*)layers->element(input_indices[l]);
		
		SGMatrix<float64_t> input = layer->get_activations();
		
		int32_t num_maps = layer->get_num_neurons()/num_neurons;
		
		for (int32_t m=0; m<num_maps; m++)
		{
			for (int32_t i=0; i<num_neurons; i++)
			{
				for (int32_t j=0; j<batch_size; j++)
				{
					buffer(i+row_offset,j) += input(i+m*num_neurons,j);
				}
			}
		}
		
		SG_UNREF(layer);
	}
	
	SGMatrix<float64_t> weights_matrix(
		parameters.vector+num_neurons, 2*m_radius_x+1, 2*m_radius_y+1, false);
	
	convolve(buffer, weights_matrix, activations, 
		false, true, row_offset, row_offset);
	
	float64_t* biases = parameters.vector;
	for (int32_t i=0; i<num_neurons; i++)
	{
		for (int32_t j=0; j<batch_size; j++)
		{
			activations(i+row_offset,j) += biases[i];
		}
	}
}

void CConvolutionalFeatureMap::compute_gradients(
	SGVector< float64_t > parameters, 
	SGMatrix< float64_t > activation_gradients, 
	CDynamicObjectArray* layers, 
	SGVector< int32_t > input_indices,
	SGVector< float64_t > parameter_gradients)
{
	int32_t num_neurons = m_width*m_height;
	int32_t batch_size = activation_gradients.num_cols;
	int32_t row_offset = m_index*num_neurons;
	
	float64_t* bias_gradients = parameter_gradients.vector;
	for (int32_t i=0; i<num_neurons; i++)
	{
		bias_gradients[i] = 0;
		for (int32_t j=0; j<batch_size; j++)
			bias_gradients[i] += activation_gradients(i+row_offset,j);
	}
	
	SGMatrix<float64_t> W(parameters.vector + num_neurons, 
		2*m_radius_x+1, 2*m_radius_y+1, false);
	
	SGMatrix<float64_t> WG(parameter_gradients.vector + num_neurons, 
		2*m_radius_x+1, 2*m_radius_y+1, false);
	
	WG.zero();
	for (int32_t l=0; l<input_indices.vlen; l++)
	{
		CNeuralLayer* layer = 
			(CNeuralLayer*)layers->element(input_indices[l]);
		
		SGMatrix<float64_t> input = layer->get_activations();
		
		int32_t num_maps = layer->get_num_neurons()/num_neurons;
		
		for (int32_t m=0; m<num_maps; m++)
		{
			compute_weight_gradients(layer->get_activations(), 
				activation_gradients, WG, m*num_neurons, row_offset);
			
			if (!layer->is_input())
				convolve(activation_gradients, W, 
					layer->get_activation_gradients(), true, false, 
					row_offset, m*num_neurons);
		}	
		
		SG_UNREF(layer);
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
			m_height, m_width, false);
		
		SGMatrix<float64_t> result(
			outputs.matrix+i*outputs.num_rows + outputs_row_offset, 
			m_height, m_width, false);
		
		for (int32_t x=0; x<m_width; x++)
		{
			for (int32_t y=0; y<m_height; y++)
			{
				float64_t sum = reset_output ? 0 : result(y,x);
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
				result(y,x) = sum;
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
	for (int32_t i=0; i<local_gradients.num_cols; i++)
	{
		SGMatrix<float64_t> image(
			inputs.matrix+i*inputs.num_rows + inputs_row_offset, 
			m_height, m_width, false);
		
		SGMatrix<float64_t> LG_image(
			local_gradients.matrix+i*local_gradients.num_rows 
			+ local_gradients_row_offset, m_height, m_width, false);
		
		for (int32_t x=0; x<m_width; x++)
		{
			for (int32_t y=0; y<m_height; y++)
			{
				for (int32_t x1=x-m_radius_x; x1<=x+m_radius_x; x1++)
				{
					for (int32_t y1=y-m_radius_y; y1<=y+m_radius_y; y1++)
					{
						if (x1>=0 && y1>=0 && x1<image.num_cols && y1<image.num_rows)
							weight_gradients(m_radius_y-y1+y,m_radius_x-x1+x) +=
								LG_image(y,x)*image(y1,x1);
					}
				}
			}
		}
	}
}
