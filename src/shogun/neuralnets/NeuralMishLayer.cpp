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
