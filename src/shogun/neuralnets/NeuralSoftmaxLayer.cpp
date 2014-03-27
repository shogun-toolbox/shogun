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

#include <shogun/neuralnets/NeuralSoftmaxLayer.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CNeuralSoftmaxLayer::CNeuralSoftmaxLayer() : CNeuralLinearLayer()
{
}

CNeuralSoftmaxLayer::CNeuralSoftmaxLayer(int32_t num_neurons): 
CNeuralLinearLayer(num_neurons)
{
}

void CNeuralSoftmaxLayer::compute_activations(float64_t* parameters,
		float64_t* previous_layer_activations)
{
	CNeuralLinearLayer::compute_activations(parameters, 
		previous_layer_activations);
	
	// to avoid exponentiating large numbers, the maximum activation is 
	// subtracted from all the activations and the computations are done in the
	// log domain
	
	float64_t max = SGVector<float64_t>::max(m_activations, m_activations.vlen);
	
	for (int32_t j=0; j<m_batch_size; j++)
	{
		float64_t sum = 0;
		for (int32_t i=0; i<m_num_neurons; i++)
		{
			sum += CMath::exp(m_activations[i+j*m_num_neurons]-max);
		}
		float64_t normalizer = CMath::log(sum);
		for (int32_t k=0; k<m_num_neurons; k++)
		{
			m_activations[k+j*m_num_neurons] =
				CMath::exp(m_activations[k+j*m_num_neurons]-max-normalizer);
		}
	}
}

void CNeuralSoftmaxLayer::compute_local_gradients(bool is_output, 
		float64_t* p)
{
	if (!is_output) SG_ERROR("%s cannot be used as a hidden layer", get_name());
	
	int32_t len = m_num_neurons*m_batch_size;
	for (int32_t i=0; i< len; i++)
	{
		m_local_gradients[i] = m_activations[i]-p[i];
	}
}

float64_t CNeuralSoftmaxLayer::computer_error(float64_t* targets)
{	
	int32_t len = m_num_neurons*m_batch_size;
	float64_t sum = 0;
	for (int32_t i=0; i< len; i++)
	{
		// to prevent taking the log of a zero
		if (m_activations[i]==0) sum += targets[i]*CMath::log(1e-50);
		else sum += targets[i]*CMath::log(m_activations[i]);
	}
	return -1*sum/m_batch_size;
}
