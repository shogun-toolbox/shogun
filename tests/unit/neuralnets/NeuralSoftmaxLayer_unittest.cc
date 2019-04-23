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

#include "NeuralLayerTestFixture.h"

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/linalg/LinalgSpecialPurposes.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralSoftmaxLayer.h>

using namespace shogun;

using NeuralSoftmaxLayerTest = NeuralLayerTestFixture;

/** Compares the activations computed using the layer against manually computed
 * activations
 */
TEST_F(NeuralSoftmaxLayerTest, compute_activations)
{
	int32_t seed = 100;
	// initialize some random inputs
	SGMatrix<float64_t> x;
	std::shared_ptr<NeuralInputLayer> input;
	std::mt19937_64 prng(seed);
	std::tie(x, input) = setup_input_layer<float64_t>(12, 3, -10.0, 10.0, prng);

	// initialize the layer
	auto layer=std::make_shared<NeuralSoftmaxLayer>(9);
	SGVector<int32_t> input_indices(1);
	input_indices[0] = 0;
	auto params =
	    init_linear_layer(layer, input_indices, x.num_cols, 1.0, false);
	SGMatrix<float64_t> A = layer->get_activations();

	// Manually compute Recitified linear activations
	auto biases =
	    SGVector<float64_t>(params.vector, layer->get_num_neurons(), 0);
	auto weights = SGMatrix<float64_t>(
	    params.vector, layer->get_num_neurons(), x.num_rows,
	    layer->get_num_neurons());
	SGMatrix<float64_t> A_ref(layer->get_num_neurons(), x.num_cols);
	shogun::linalg::add_vector(
	    shogun::linalg::matrix_prod(weights, x), biases, A_ref);
	shogun::linalg::softmax(A_ref);

	// compare
	EXPECT_EQ(A_ref.num_rows, A.num_rows);
	EXPECT_EQ(A_ref.num_cols, A.num_cols);
	for (int32_t i = 0; i < A.num_rows * A.num_cols; i++)
		EXPECT_NEAR(A_ref[i], A[i], 1e-12);
}

/** Compares the error computed using the layer against a manually computed
 * error
 */
TEST_F(NeuralSoftmaxLayerTest, compute_error)
{
	int32_t seed = 100;
	// initialize some random inputs
	SGMatrix<float64_t> x;
	std::shared_ptr<NeuralInputLayer> input;
	std::mt19937_64 prng(seed);
	std::tie(x, input) = setup_input_layer<float64_t>(12, 3, -10.0, 10.0, prng);

	// initialize the softmax layer
	auto layer=std::make_shared<NeuralSoftmaxLayer>(9);
	layer->put("seed", seed);
	SGVector<int32_t> input_indices(1);
	input_indices[0] = 0;
	auto params =
	    init_linear_layer(layer, input_indices, x.num_cols, 1.0, false);

	// initialize the output
	auto y = create_rand_matrix<float64_t>(
	    layer->get_num_neurons(), x.num_cols, 0.0, 1.0, prng);

	// make sure y is in the form of a probability distribution
	auto sum_vect = shogun::linalg::colwise_sum(y);

	for (int32_t j = 0; j < y.num_cols; j++)
	{
		auto y_j = y.get_column(j);
		shogun::linalg::scale(y_j, y_j, 1.0 / sum_vect[j]);
		y.set_column(j, y_j);
	}

	// compute the layer's local gradients
	layer->compute_local_gradients(y);
	SGMatrix<float64_t> LG = layer->get_local_gradients();

	// manually compute local gradients
	SGMatrix<float64_t> A = layer->get_activations();
	SGMatrix<float64_t> LG_ref(A.num_rows, A.num_cols);

	for (int32_t i = 0; i < A.num_rows * A.num_cols; i++)
		LG_ref[i] = (A[i] - y[i]) / x.num_cols;

	// compare
	EXPECT_EQ(LG_ref.num_rows, LG.num_rows);
	EXPECT_EQ(LG_ref.num_cols, LG.num_cols);
	for (int32_t i = 0; i < LG.num_rows * LG.num_cols; i++)
		EXPECT_NEAR(LG_ref[i], LG[i], 1e-6);
}
