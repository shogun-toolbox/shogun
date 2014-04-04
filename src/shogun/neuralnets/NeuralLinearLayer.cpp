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

#include <shogun/neuralnets/NeuralLinearLayer.h>
#include <shogun/mathematics/Random.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif

using namespace shogun;

CNeuralLinearLayer::CNeuralLinearLayer() : CNeuralLayer()
{
}

CNeuralLinearLayer::CNeuralLinearLayer(int32_t num_neurons): 
CNeuralLayer(num_neurons)
{	
}

int32_t CNeuralLinearLayer::get_num_parameters()
{
	// A weight matrix of size num_neurons*previous_layer_num_neurons and vector
	// of biases of size num_neurons
	return m_num_neurons*m_previous_layer_num_neurons + m_num_neurons;
}

void CNeuralLinearLayer::initialize_parameters(float64_t* parameters, 
		bool* parameter_regularizable, 
		float64_t sigma)
{
	CRandom random_generator(CRandom::generate_seed());
	
	// random the parameters
	int32_t num_parameters = get_num_parameters();
	for (int32_t i=0; i<num_parameters; i++)
		parameters[i] = random_generator.normal_distrib(0.0, sigma);
	
	// turn regularization off for the biases, on for the weights
	int32_t num_weights = m_num_neurons*m_previous_layer_num_neurons;
	for (int32_t i=0; i<num_parameters; i++)
		parameter_regularizable[i] = (i<num_weights);
}


void CNeuralLinearLayer::compute_activations(float64_t* parameters, 
		float64_t* previous_layer_activations)
{
	float64_t* weights = parameters;
	float64_t* biases = parameters + m_num_neurons*m_previous_layer_num_neurons;
	
#ifdef HAVE_EIGEN3
	typedef Eigen::Map<Eigen::MatrixXd> EMappedMatrix;
	typedef Eigen::Map<Eigen::VectorXd> EMappedVector;
	
	EMappedMatrix W(weights, m_num_neurons, 
			m_previous_layer_num_neurons);
	EMappedMatrix X(previous_layer_activations, 
			m_previous_layer_num_neurons, m_batch_size);
	EMappedMatrix  A(m_activations, m_num_neurons, m_batch_size);
	EMappedVector  B(biases, m_num_neurons);
	
	A = W*X;
	A.colwise() += B;
#else
	// activations = weights*previous_layer_activations
	for (int32_t i=0; i<m_num_neurons; i++)
	{
		for (int32_t j=0; j<m_batch_size; j++)
		{
			float64_t sum = 0;
			for (int32_t k=0; k<m_previous_layer_num_neurons; k++)
			{
				sum += weights[i+k*m_num_neurons]*
					previous_layer_activations[k+j*m_previous_layer_num_neurons]
					;
			}
			m_activations[i+j*m_num_neurons] = sum;
		}
	}
	
	// add biases
	for (int32_t i=0; i<m_num_neurons; i++)
	{
		for (int32_t j=0; j<m_batch_size; j++)
		{
			m_activations[i+j*m_num_neurons] += biases[i];
		}
	}
	
#endif
}

void CNeuralLinearLayer::compute_gradients(float64_t* parameters,
		bool is_output, float64_t* p,
		float64_t* previous_layer_activations,
		float64_t* parameter_gradients)
{
	float64_t* weights = parameters;
	
	compute_local_gradients(is_output, p);
	
#ifdef HAVE_EIGEN3
	typedef Eigen::Map<Eigen::MatrixXd> EMappedMatrix;
	typedef Eigen::Map<Eigen::VectorXd> EMappedVector;
	
	EMappedMatrix X(previous_layer_activations, 
			m_previous_layer_num_neurons, m_batch_size);
	EMappedMatrix  W(weights, m_num_neurons, 
			m_previous_layer_num_neurons);
	EMappedMatrix LG(m_local_gradients, m_num_neurons, 
			m_batch_size);
	EMappedMatrix WG(parameter_gradients, 
			m_num_neurons, m_previous_layer_num_neurons);
	EMappedVector BG(parameter_gradients + 
				m_num_neurons*m_previous_layer_num_neurons, m_num_neurons);
	EMappedMatrix  IG(m_input_gradients, 
			   m_previous_layer_num_neurons, m_batch_size);
	
	// compute parameter gradients
	WG = (LG*X.transpose())/m_batch_size;
	BG = LG.rowwise().sum()/m_batch_size;
	
	// compute input gradients
	IG = W.transpose()*LG;
#else
	float64_t* weight_gradients = parameter_gradients;
	float64_t* bias_gradients = parameter_gradients + 
		m_num_neurons*m_previous_layer_num_neurons;
		
	// weight_gradients=local_gradients*previous_layer_activations.T/batch_size
	for (int32_t i=0; i<m_num_neurons; i++)
	{
		for (int32_t j=0; j<m_previous_layer_num_neurons; j++)
		{
			float64_t sum = 0;
			for (int32_t k=0; k<m_batch_size; k++)
			{
				sum += m_local_gradients[i+k*m_num_neurons]*
					previous_layer_activations[j+k*m_previous_layer_num_neurons]
					;
			}
			weight_gradients[i+j*m_num_neurons] = sum/m_batch_size;
		}
	}
	
	// bias_gradients = local_gradients.row_sum()/batch_size
	for (int32_t i=0; i<m_num_neurons; i++)
	{
		float64_t sum = 0;
		for (int32_t j=0; j<m_batch_size; j++)
		{
			sum += m_local_gradients[i+j*m_num_neurons];
		}
		bias_gradients[i] = sum/m_batch_size;
	}
	
	// input_gradients = weights.T*local_gradients
	for (int32_t i=0; i<m_previous_layer_num_neurons; i++)
	{
		for (int32_t j=0; j<m_batch_size; j++)
		{
			float64_t sum = 0;
			for (int32_t k=0; k<m_num_neurons; k++)
			{
				sum += weights[k+i*m_num_neurons]*
					m_local_gradients[k+j*m_num_neurons];
			}
			m_input_gradients[i+j*m_previous_layer_num_neurons] = sum;
		}
	}
#endif
}

void CNeuralLinearLayer::compute_local_gradients(bool is_output, float64_t* p)
{	
	if (is_output)
	{
		// sqaured error measure
		// local_gradients = activations-targets
		int32_t length = m_num_neurons*m_batch_size;
		for (int32_t i=0; i<length; i++)
			m_local_gradients[i] = m_activations[i]-p[i];
		
	}
	else
	{
		int32_t length = m_num_neurons*m_batch_size;
		for (int32_t i=0; i<length; i++)
			m_local_gradients[i] = p[i];
	}
	
}

float64_t CNeuralLinearLayer::compute_error(float64_t* targets)
{
	// error = 0.5*(sum(targets-activations)^2)/batch_size
	float64_t sum = 0;
	int32_t length = m_num_neurons*m_batch_size;
	for (int32_t i=0; i<length; i++)
		sum += (targets[i]-m_activations[i])*(targets[i]-m_activations[i]);
	sum *= (0.5/m_batch_size);
	return sum;
}
