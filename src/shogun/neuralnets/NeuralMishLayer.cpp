/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors : Manjunath Bhat
 */
#include <shogun/neuralnets/NeuralMishLayer.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGVector.h>

using namespace shogun;

NeuralMishLayer::NeuralMishLayer() : NeuralLinearLayer()
{
}

NeuralMishLayer::NeuralMishLayer(int32_t num_neurons):
NeuralLinearLayer(num_neurons)
{
}

void NeuralMishLayer::compute_activations(SGVector<float64_t> parameters,
    const std::vector<std::shared_ptr<NeuralLayer>>& layers)
{
	NeuralLinearLayer::compute_activations(parameters, layers);

	for(auto& activation: m_activations)
	{
		activation *= std::tanh(std::log(1 + std::exp(activation)));
	}
}

void NeuralMishLayer::compute_local_gradients(
		SGMatrix<float64_t> targets)
{
	if (targets.num_rows != 0)
	{
		int32_t length = m_num_neurons*m_batch_size;
		for (int32_t i=0; i<length; i++)
		{
			m_local_gradients[i] = (m_activations[i]-targets[i])/m_batch_size;
		}
	}
	else
	{
		int32_t len = m_num_neurons*m_batch_size;
		for (int32_t i=0; i< len; i++)
		{
			auto e = std::exp(m_activations[i]);
			auto omega = 4 * (m_activations[i] + 1) + 4*pow(e, 2) + pow(e, 3) + e*(4*m_activations[i] + 6);
			auto delta = 2*e + pow(e, 2) + 2;
			m_local_gradients[i] = m_activation_gradients[i] * e * omega/pow(delta, 2);
		}
	}
}
