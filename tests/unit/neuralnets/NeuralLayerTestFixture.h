/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saatvik Shah
 */

#ifndef NEURAL_LAYER_TEST_FIXTURE_H
#define NEURAL_LAYER_TEST_FIXTURE_H

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralLinearLayer.h>
#include <shogun/mathematics/UniformRealDistribution.h>
#include <shogun/mathematics/UniformIntDistribution.h>

#include <gtest/gtest.h>

#include <memory>
#include <tuple>
#include <type_traits>

using namespace shogun;

class NeuralLayerTestFixture : public ::testing::Test
{
public:
	template <typename T, typename PRNG>
	auto create_rand_matrix(
	    int32_t num_rows, int32_t num_cols, T lower_bound, T upper_bound, PRNG& prng)
	{
		auto data_batch = SGMatrix<T>(num_rows, num_cols);
		if constexpr (std::is_floating_point<T>::value)
		{
			UniformRealDistribution<T> uniform_real_dist(lower_bound, upper_bound);
			for (size_t i = 0; i < num_rows * num_cols; i++)
			{
				data_batch[i] = uniform_real_dist(prng);
			}
		}
		else if constexpr (std::is_integral<T>::value)
		{
			UniformIntDistribution<T> uniform_int_dist(lower_bound, upper_bound);
			for (size_t i = 0; i < num_rows * num_cols; i++)
			{
				data_batch[i] = uniform_int_dist(prng);
			}
		}
		return data_batch;
	}

	/**
	 * Generates a random dataset according to provided specifications.
	 * Then performs the boilerplate needed to setup the input layer
	 * for this dataset.
	 * @tparam T: Type of input data eg. float64_t
	 * @param num_features: For the randomized dataset
	 * @param num_samples: For the randomized dataset
	 * @param lower_bound/upper_bound: Dataset generated has values between:
	 * [lower_bound, upper_bound]
	 * @param add_to_layers: Should this layer be added to the dynamic layer
	 * list(`m_layers`)
	 * This is normally done when expecting to connect this layer ahead to some
	 * other layer
	 * @return: The randomized dataset and corresponding Neural input layer
	 */
	template <typename T, typename PRNG>
	auto setup_input_layer(
	    int32_t num_features, int32_t num_samples, T lower_bound, T upper_bound,
	    PRNG& prng, bool add_to_layers = true)
	{
		auto data_batch = create_rand_matrix(
		    num_features, num_samples, lower_bound, upper_bound, prng);
		auto input_layer = std::make_shared<NeuralInputLayer>(data_batch.num_rows);
		input_layer->set_batch_size(data_batch.num_cols);
		input_layer->compute_activations(data_batch);
		if (add_to_layers)
		{
			m_layers.push_back(input_layer);
		}
		return std::make_tuple(data_batch, input_layer);
	}

	/**
	 * Initializes Linear layer metadata according to specifications provided
	 * @param layer: The layer to initialize
	 * @param input_indices: the indices of layers from `m_layers` that are
	 * inputs to this layer
	 * @param batch_size: Batch size for this layer
	 * @param sigma: The parameters are initialized as Normal(0 mean, sigma
	 * stdev)
	 * @param add_to_layers: Whether this layer should be added to the dynamic
	 * layer list(`m_layers`)
	 * This is normally done when expecting to connect this layer ahead to some
	 * other layer.
	 * @return: The initialized parameters
	 */
	auto init_linear_layer(
	    std::shared_ptr<NeuralLinearLayer> layer, const SGVector<int32_t>& input_indices,
	    int32_t batch_size, double sigma, bool add_to_layers)
	{
		if (add_to_layers)
		{
			m_layers.push_back(layer);
		}
		layer->initialize_neural_layer(m_layers, input_indices);
		SGVector<float64_t> params(layer->get_num_parameters());
		SGVector<bool> param_regularizable(layer->get_num_parameters());
		layer->initialize_parameters(params, param_regularizable, sigma);
		layer->set_batch_size(batch_size);
		layer->compute_activations(params, m_layers);
		return params;
	}

	// dynamic list of layers
	std::vector<std::shared_ptr<NeuralLayer>> m_layers;

protected:
	void SetUp() final
	{
		m_layers.clear();
	}
};
#endif // NEURAL_LAYER_TEST_FIXTURE_H
