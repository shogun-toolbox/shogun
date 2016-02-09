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
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGVector.h>

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

void CNeuralLinearLayer::initialize_neural_layer(CDynamicObjectArray* layers,
		SGVector< int32_t > input_indices)
{
	CNeuralLayer::initialize_neural_layer(layers, input_indices);

	m_num_parameters = m_num_neurons;
	for (int32_t i=0; i<input_indices.vlen; i++)
		m_num_parameters += m_num_neurons*m_input_sizes[i];
}

void CNeuralLinearLayer::initialize_parameters(SGVector<float64_t> parameters,
		SGVector<bool> parameter_regularizable,
		float64_t sigma)
{
	for (int32_t i=0; i<m_num_parameters; i++)
	{
		// random the parameters
		parameters[i] = CMath::normal_random(0.0, sigma);

		// turn regularization off for the biases, on for the weights
		parameter_regularizable[i] = (i>=m_num_neurons);
	}
}

void CNeuralLinearLayer::compute_activations(SGVector<float64_t> parameters,
		CDynamicObjectArray* layers)
{
	float64_t* biases = parameters.vector;

#ifdef HAVE_EIGEN3
	typedef Eigen::Map<Eigen::MatrixXd> EMappedMatrix;
	typedef Eigen::Map<Eigen::VectorXd> EMappedVector;

	EMappedMatrix  A(m_activations.matrix, m_num_neurons, m_batch_size);
	EMappedVector  B(biases, m_num_neurons);

	A.colwise() = B;
#else
	for (int32_t i=0; i<m_num_neurons; i++)
	{
		for (int32_t j=0; j<m_batch_size; j++)
		{
			m_activations[i+j*m_num_neurons] = biases[i];
		}
	}
#endif

	int32_t weights_index_offset = m_num_neurons;
	for (int32_t l=0; l<m_input_indices.vlen; l++)
	{
		CNeuralLayer* layer =
			(CNeuralLayer*)layers->element(m_input_indices[l]);

		float64_t* weights = parameters.vector + weights_index_offset;
		weights_index_offset += m_num_neurons*layer->get_num_neurons();

#ifdef HAVE_EIGEN3
		EMappedMatrix W(weights, m_num_neurons, layer->get_num_neurons());
		EMappedMatrix X(layer->get_activations().matrix,
				layer->get_num_neurons(), m_batch_size);

		A += W*X;
#else
		// activations = weights*previous_layer_activations
		for (int32_t i=0; i<m_num_neurons; i++)
		{
			for (int32_t j=0; j<m_batch_size; j++)
			{
				float64_t sum = 0;
				for (int32_t k=0; k<layer->get_num_neurons(); k++)
				{
					sum += weights[i+k*m_num_neurons]*
						layer->get_activations()(k,j);
				}
				m_activations[i+j*m_num_neurons] += sum;
			}
		}
#endif
		SG_UNREF(layer);
	}
}

void CNeuralLinearLayer::compute_gradients(
		SGVector<float64_t> parameters,
		SGMatrix<float64_t> targets,
		CDynamicObjectArray* layers,
		SGVector<float64_t> parameter_gradients)
{
	compute_local_gradients(targets);

	// compute bias gradients
	float64_t* bias_gradients = parameter_gradients.vector;
#ifdef HAVE_EIGEN3
	typedef Eigen::Map<Eigen::MatrixXd> EMappedMatrix;
	typedef Eigen::Map<Eigen::VectorXd> EMappedVector;

	EMappedVector BG(bias_gradients, m_num_neurons);
	EMappedMatrix LG(m_local_gradients.matrix, m_num_neurons, m_batch_size);

	BG = LG.rowwise().sum();
#else
	for (int32_t i=0; i<m_num_neurons; i++)
	{
		float64_t sum = 0;
		for (int32_t j=0; j<m_batch_size; j++)
		{
			sum += m_local_gradients[i+j*m_num_neurons];
		}
		bias_gradients[i] = sum;
	}
#endif

	// apply dropout to the local gradients
	if (dropout_prop>0.0)
	{
		int32_t len = m_num_neurons*m_batch_size;
		for (int32_t i=0; i<len; i++)
			m_local_gradients[i] *= m_dropout_mask[i];
	}

	int32_t weights_index_offset = m_num_neurons;
	for (int32_t l=0; l<m_input_indices.vlen; l++)
	{
		CNeuralLayer* layer =
			(CNeuralLayer*)layers->element(m_input_indices[l]);

		float64_t* weights = parameters.vector + weights_index_offset;
		float64_t* weight_gradients = parameter_gradients.vector +
			weights_index_offset;

		weights_index_offset += m_num_neurons*layer->get_num_neurons();

#ifdef HAVE_EIGEN3
		EMappedMatrix X(layer->get_activations().matrix,
				layer->get_num_neurons(), m_batch_size);
		EMappedMatrix  W(weights, m_num_neurons, layer->get_num_neurons());
		EMappedMatrix WG(weight_gradients,
				m_num_neurons, layer->get_num_neurons());
		EMappedMatrix  IG(layer->get_activation_gradients().matrix,
				layer->get_num_neurons(), m_batch_size);

		// compute weight gradients
		WG = LG*X.transpose();

		// compute input gradients
		if (!layer->is_input())
			IG += W.transpose()*LG;
#else
		// weight_gradients=local_gradients*previous_layer_activations.T
		for (int32_t i=0; i<m_num_neurons; i++)
		{
			for (int32_t j=0; j<layer->get_num_neurons(); j++)
			{
				float64_t sum = 0;
				for (int32_t k=0; k<m_batch_size; k++)
				{
					sum += m_local_gradients(i,k)*layer->get_activations()(j,k);
				}
				weight_gradients[i+j*m_num_neurons] = sum;
			}
		}

		if (!layer->is_input())
		{
			// input_gradients = weights.T*local_gradients
			for (int32_t i=0; i<layer->get_num_neurons(); i++)
			{
				for (int32_t j=0; j<m_batch_size; j++)
				{
					float64_t sum = 0;
					for (int32_t k=0; k<m_num_neurons; k++)
					{
						sum += weights[k+i*m_num_neurons]*
							m_local_gradients[k+j*m_num_neurons];
					}
					layer->get_activation_gradients()(i,j) += sum;
				}
			}
		}
#endif
		SG_UNREF(layer);
	}

	if (contraction_coefficient != 0)
	{
		compute_contraction_term_gradients(parameters, parameter_gradients);
	}
}

void CNeuralLinearLayer::compute_local_gradients(SGMatrix<float64_t> targets)
{
	if (targets.num_rows != 0)
	{
		// sqaured error measure
		// local_gradients = activations-targets
		int32_t length = m_num_neurons*m_batch_size;
		for (int32_t i=0; i<length; i++)
			m_local_gradients[i] = (m_activations[i]-targets[i])/m_batch_size;
	}
	else
	{
		int32_t length = m_num_neurons*m_batch_size;
		for (int32_t i=0; i<length; i++)
			m_local_gradients[i] = m_activation_gradients[i];
	}
}

float64_t CNeuralLinearLayer::compute_error(SGMatrix<float64_t> targets)
{
	// error = 0.5*(sum(targets-activations)^2)/batch_size
	float64_t sum = 0;
	int32_t length = m_num_neurons*m_batch_size;
	for (int32_t i=0; i<length; i++)
		sum += (targets[i]-m_activations[i])*(targets[i]-m_activations[i]);
	sum *= (0.5/m_batch_size);
	return sum;
}

void CNeuralLinearLayer::enforce_max_norm(SGVector<float64_t> parameters,
		float64_t max_norm)
{
	int32_t weights_index_offset = m_num_neurons;
	for (int32_t l=0; l<m_input_indices.vlen; l++)
	{
		float64_t* weights = parameters.vector + weights_index_offset;

		int32_t length = m_num_neurons*m_input_sizes[l];
		for (int32_t i=0; i<length; i+=m_input_sizes[l])
		{
			float64_t norm =
				SGVector<float64_t>::twonorm(parameters.vector+i, m_num_neurons);

			if (norm > max_norm)
			{
				float64_t multiplier = max_norm/norm;
				for (int32_t j=0; j<m_input_sizes[l]; j++)
					weights[i+j] *= multiplier;
			}
		}
	}
}

float64_t CNeuralLinearLayer::compute_contraction_term(SGVector<float64_t> parameters)
{
	float64_t contraction_term = 0;
	for (int32_t i=m_num_neurons; i<parameters.vlen; i++)
		contraction_term += parameters[i]*parameters[i];

	return contraction_coefficient*contraction_term;
}

void CNeuralLinearLayer::compute_contraction_term_gradients(
	SGVector< float64_t > parameters, SGVector< float64_t > gradients)
{
	for (int32_t i=m_num_neurons; i<parameters.vlen; i++)
			gradients[i] += 2*contraction_coefficient*parameters[i];
}

