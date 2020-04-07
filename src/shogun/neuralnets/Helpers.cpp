/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/features/DenseFeatures.h>
#include <shogun/neuralnets/DeepAutoencoder.h>
#include <shogun/neuralnets/Helpers.h>

using namespace shogun;

std::shared_ptr<DenseFeatures<float64_t>> shogun::reconstruct(const std::shared_ptr<Machine>& net, 
		const std::shared_ptr<Features>& data)
{
	auto dense_features = data->as<DenseFeatures<float64_t>>();
	require(dense_features, "Expected dense features.");

	if (auto ae = net->as<DeepAutoencoder>())
		return ae->reconstruct(dense_features);
	if (auto ae = net->as<Autoencoder>())
		return ae->reconstruct(dense_features);
	error("Expected net to be a DeepAutoencoder or Autoencoder.");
}

std::shared_ptr<Machine> shogun::convert_to_neural_network(const std::shared_ptr<Machine>& net, 
		std::shared_ptr<NeuralLayer> output_layer, float64_t sigma)
{
	if (auto ae = net->as<DeepAutoencoder>())
		return ae->convert_to_neural_network(output_layer, sigma);
	error("Expected net to be a DeepAutoencoder.");
}