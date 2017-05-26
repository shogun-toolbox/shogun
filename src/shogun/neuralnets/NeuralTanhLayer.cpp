/*
 * Copyright (c) 2016, Shogun Toolbox Foundation
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
 * Written (W) 2016 Arasu Arun
 */

#include <shogun/neuralnets/NeuralTanhLayer.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGVector.h>

using namespace shogun;

CNeuralTanhLayer::CNeuralTanhLayer() : CNeuralLinearLayer()
{
}

CNeuralTanhLayer::CNeuralTanhLayer(int32_t num_neurons):
CNeuralLinearLayer(num_neurons)
{
}

void CNeuralTanhLayer::compute_activations(SGVector<float64_t> parameters,
		CDynamicObjectArray* layers)
{
	CNeuralLinearLayer::compute_activations(parameters, layers);

	// apply tanh activation function
	int32_t length = m_num_neurons*m_batch_size;
	for (int32_t i=0; i<length; i++)
		m_activations[i] = CMath::exp(-2.0*m_activations[i]));
		m_activations[i] = (1.0-m_activations[i])/(1.0+m_activations[i]);
}

float64_t CNeuralTanhLayer::compute_contraction_term(
	SGVector< float64_t > parameters)
{
	int32_t num_inputs = SGVector<int32_t>::sum(m_input_sizes.vector, m_input_sizes.vlen);

	SGMatrix<float64_t> W(parameters.vector+m_num_neurons,
		m_num_neurons, num_inputs, false);

	float64_t contraction_term = 0;
	for (int32_t i=0; i<m_num_neurons; i++)
	{
		float64_t sum_j = 0;
		for (int32_t j=0; j<num_inputs; j++)
			sum_j += W(i,j)*W(i,j);

		for (int32_t k=0; k<m_batch_size; k++)
		{
			float64_t h_ = 1-m_activations(i,k)*m_activations(i,k);
			contraction_term += h_*h_*sum_j;
		}
	}

	return (contraction_coefficient/m_batch_size) * contraction_term;
}

void CNeuralTanhLayer::compute_contraction_term_gradients(
	SGVector< float64_t > parameters, SGVector< float64_t > gradients)
{
	int32_t num_inputs = SGVector<int32_t>::sum(m_input_sizes.vector, m_input_sizes.vlen);

	SGMatrix<float64_t> W(parameters.vector+m_num_neurons,
		m_num_neurons, num_inputs, false);
	SGMatrix<float64_t> WG(gradients.vector+m_num_neurons,
		m_num_neurons, num_inputs, false);

	for (int32_t k = 0; k<m_batch_size; k++)
	{
		for (int32_t i=0; i<m_num_neurons; i++)
		{
			float64_t sum_j = 0;
			for (int32_t j=0; j<num_inputs; j++)
				sum_j += W(i,j)*W(i,j);

			for (int32_t j=0; j<num_inputs; j++)
			{
				float64_t h = m_activations(i,k);
				float64_t w = W(i,j);
				float64_t h_ = 1-m_activations(i,k)*m_activations(i,k);

				float64_t g = 2*h_*h_*(w-2*h*sumj);

				WG(i,j) += (contraction_coefficient/m_batch_size)*g;
			}
		}
	}
}


void CNeuralTanhLayer::compute_local_gradients(SGMatrix<float64_t> targets)
{
	CNeuralLinearLayer::compute_local_gradients(targets);

	// multiply by the derivative of the tanh function
	int32_t length = m_num_neurons*m_batch_size;
	for (int32_t i=0; i<length; i++)
		m_local_gradients[i] *= 1.0-m_activations[i]*m_activations[i];
}
