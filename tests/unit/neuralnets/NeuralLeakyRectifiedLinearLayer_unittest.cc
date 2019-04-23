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
 * Written (W) 2015 Sanuj Sharma
 */
#include "NeuralLayerTestFixture.h"
#include <gtest/gtest.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralLeakyRectifiedLinearLayer.h>

using namespace shogun;

using NeuralLeakyRectifiedLayerTest = NeuralLayerTestFixture;

/** Compares the activations computed using the layer against manually computed
 * activations
 */
TEST_F(NeuralLeakyRectifiedLayerTest, compute_activations)
{
	int32_t seed = 100;
	// initialize some random inputs
	SGMatrix<float64_t> x;
	std::shared_ptr<NeuralInputLayer> input;
	std::mt19937_64 prng(seed);
	std::tie(x, input) = setup_input_layer<float64_t>(12, 3, -10.0, 10.0, prng);

	// initialize leaky rectified linear layer
	auto layer = std::make_shared<NeuralLeakyRectifiedLinearLayer>(9);
	layer->set_alpha(0.02);
	SGVector<int32_t> input_indices(1);
	input_indices[0] = 0;
	auto params =
	    init_linear_layer(layer, input_indices, x.num_cols, 1.0, false);

	// get the layer's activations
	SGMatrix<float64_t> A = layer->get_activations();

	// manually compute the layer's activations
	auto biases =
	    SGVector<float64_t>(params.vector, layer->get_num_neurons(), 0);
	auto weights = SGMatrix<float64_t>(
	    params.vector, layer->get_num_neurons(), x.num_rows,
	    layer->get_num_neurons());
	SGMatrix<float64_t> A_ref(layer->get_num_neurons(), x.num_cols);
	shogun::linalg::add_vector(
	    shogun::linalg::matrix_prod(weights, x), biases, A_ref);

	for (int32_t idx = 0; idx < A_ref.num_rows * A_ref.num_cols; idx++)
	{
		A_ref[idx] =
		    Math::max<float64_t>(layer->get_alpha() * A_ref[idx], A_ref[idx]);
	}

	// compare
	EXPECT_EQ(A_ref.num_rows, A.num_rows);
	EXPECT_EQ(A_ref.num_cols, A.num_cols);
	for (int32_t i = 0; i < A.num_rows * A.num_cols; i++)
	{
		EXPECT_NEAR(A_ref[i], A[i], 1e-12);
	}
}
