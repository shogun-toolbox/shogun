/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors : Manjunath Bhat
 */
#include "NeuralLayerTestFixture.h"
#include <gtest/gtest.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralMishLayer.h>
#include <shogun/mathematics/UniformIntDistribution.h>

using namespace shogun;

using NeuralMishLayerTest = NeuralLayerTestFixture;

/** Compares the activations computed using the layer against manually computed
 * activations
 */
TEST_F(NeuralMishLayerTest, compute_activations)
{
	int32_t seed = 100;
	// initialize some random inputs
	SGMatrix<float64_t> x;
	std::shared_ptr<NeuralInputLayer> input;
	std::mt19937_64 prng(seed);
	auto [x, input] = setup_input_layer<float64_t>(12, 3, -10.0, 10.0, prng);

	// initialize mish layer
	auto layer = std::make_shared<NeuralMishLayer>(9);
	SGVector<int32_t> input_indices(1);
	input_indices[0] = 0;
	auto params =
	    init_linear_layer(layer, input_indices, x.num_cols, 1.0, false);

	// get the layer's activations
	SGMatrix<float64_t> A = layer->get_activations();

	auto num_neurons = layer->get_num_neurons();
	auto num_rows = x.num_rows;
	auto num_cols = x.num_cols;
	// manually compute the layer's activations
	auto biases =
	    SGVector<float64_t>(params.vector, num_neurons, 0);
	auto weights = SGMatrix<float64_t>(
	    params.vector, num_neurons, num_rows,
	    num_neurons);
	SGMatrix<float64_t> A_ref(num_neurons, num_cols);
	shogun::linalg::add_vector(
	    shogun::linalg::matrix_prod(weights, x), biases, A_ref);

	for (int32_t idx = 0; idx < A_ref.num_rows * A_ref.num_cols; idx++)
	{
		A_ref[idx] = A_ref[idx] * std::tanh(std::log(1 + std::exp(A_ref[idx])));
	}

	// compare
	EXPECT_EQ(A_ref.num_rows, A.num_rows);
	EXPECT_EQ(A_ref.num_cols, A.num_cols);
	for (int32_t i = 0; i < A.num_rows * A.num_cols; i++)
	{
		EXPECT_NEAR(A_ref[i], A[i], std::numeric_limits<float64_t>::epsilon);
	}
}
