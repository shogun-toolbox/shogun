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

#include "shogun/neuralnets/NeuralSoftmaxLayer.h"
#include "shogun/lib/SGMatrix.h"
#include "shogun/lib/SGVector.h"
#include "shogun/mathematics/Math.h"
#include "shogun/neuralnets/NeuralInputLayer.h"
#include "utils/Utils.h"
#include "gtest/gtest.h"

using namespace shogun;

class NeuralSoftmaxLayerTest : public ::testing::Test
{
protected:
	void SetUp()
	{
		CMath::init_random(100);
		m_layers = std::make_unique<CDynamicObjectArray>();
	}
	std::unique_ptr<CDynamicObjectArray> m_layers;
};

/** Compares the activations computed using the layer against manually computed
 * activations
 */
TEST_F(NeuralSoftmaxLayerTest, compute_activations)
{
	// initialize some random inputs
	SGMatrix<float64_t> x;
	CNeuralInputLayer* input;
	std::tie(x, input) =
	    NeuralLayerTestUtil::create_rand_input_layer<float64_t>(
	        12, 3, -10.0, 10.0);
	m_layers->append_element(input);

	// initialize the layer
	CNeuralSoftmaxLayer layer(9);
	SGVector<int32_t> input_indices(1);
	input_indices[0] = 0;
	auto params = NeuralLayerTestUtil::init_neural_linear_layer(
	    &layer, m_layers.get(), input_indices, x.num_cols, 1.0);
	SGMatrix<float64_t> A = layer.get_activations();

	// manually compute the layer's activations
	SGMatrix<float64_t> A_ref(layer.get_num_neurons(), x.num_cols);

	float64_t* biases = params.vector;
	float64_t* weights = biases + layer.get_num_neurons();

	for (int32_t i=0; i<A_ref.num_rows; i++)
	{
		for (int32_t j=0; j<A_ref.num_cols; j++)
		{
			A_ref(i,j) = biases[i];

			for (int32_t k=0; k<x.num_rows; k++)
				A_ref(i,j) += weights[i+k*A_ref.num_rows]*x(k,j);

			A_ref(i, j) = std::exp(A_ref(i, j));
		}
	}

	for (int32_t j=0; j<A_ref.num_cols; j++)
	{
		float64_t sum = 0;
		for (int32_t k=0; k<A_ref.num_rows; k++)
			sum += A_ref(k,j);

		for (int32_t i=0; i<A_ref.num_rows; i++)
			A_ref(i,j) /= sum;
	}

	// compare
	EXPECT_EQ(A_ref.num_rows, A.num_rows);
	EXPECT_EQ(A_ref.num_cols, A.num_cols);
	for (int32_t i=0; i<A.num_rows*A.num_cols; i++)
		EXPECT_NEAR(A_ref[i], A[i], 1e-12);
}

/** Compares the error computed using the layer against a manually computed
 * error
 */
TEST_F(NeuralSoftmaxLayerTest, compute_error)
{
	// initialize some random inputs
	SGMatrix<float64_t> x;
	CNeuralInputLayer* input;
	std::tie(x, input) =
	    NeuralLayerTestUtil::create_rand_input_layer<float64_t>(
	        12, 3, -10.0, 10.0);
	m_layers->append_element(input);

	// initialize the softmax layer
	CNeuralSoftmaxLayer layer(9);
	SGVector<int32_t> input_indices(1);
	input_indices[0] = 0;
	auto params = NeuralLayerTestUtil::init_neural_linear_layer(
	    &layer, m_layers.get(), input_indices, x.num_cols, 1.0);

	// initialize output
	auto y = NeuralLayerTestUtil::create_rand_sgmat<float64_t>(9, 3, 0.0, 1.0);

	// make sure y is in the form of a probability distribution
	for (int32_t j=0; j<y.num_cols; j++)
	{
		float64_t sum = 0;
		for (int32_t k=0; k<y.num_rows; k++)
			sum += y(k,j);

		for (int32_t i=0; i<y.num_rows; i++)
			y(i,j) /= sum;
	}

	// compute the layer's activations and error
	SGMatrix<float64_t> A = layer.get_activations();
	float64_t error = layer.compute_error(y);

	// manually compute error
	float64_t error_ref = 0;
	for (int32_t i=0; i<A.num_rows*A.num_cols; i++)
		error_ref += y[i] * CMath::max(std::log(1e-50), std::log(A[i]));
	error_ref *= -1.0/x.num_cols;

	// compare
	EXPECT_NEAR(error_ref, error, 1e-12);
}

/** Compares the local gradients computed using the layer against manually
 * computed gradients
 */
TEST_F(NeuralSoftmaxLayerTest, compute_local_gradients)
{
	// initialize some random inputs
	SGMatrix<float64_t> x;
	CNeuralInputLayer* input;
	std::tie(x, input) =
	    NeuralLayerTestUtil::create_rand_input_layer<float64_t>(
	        12, 3, -10.0, 10.0);
	m_layers->append_element(input);

	// initialize the layer
	CNeuralSoftmaxLayer layer(9);
	SGVector<int32_t> input_indices(1);
	input_indices[0] = 0;
	auto params = NeuralLayerTestUtil::init_neural_linear_layer(
	    &layer, m_layers.get(), input_indices, x.num_cols, 1.0);

	// initialize the output
	auto y = NeuralLayerTestUtil::create_rand_sgmat<float64_t>(
	    layer.get_num_neurons(), x.num_cols, 0.0, 1.0);

	// make sure y is in the form of a probability distribution
	for (int32_t j=0; j<y.num_cols; j++)
	{
		float64_t sum = 0;
		for (int32_t k=0; k<y.num_rows; k++)
			sum += y(k,j);

		for (int32_t i=0; i<y.num_rows; i++)
			y(i,j) /= sum;
	}

	// compute the layer's local gradients
	layer.compute_local_gradients(y);
	SGMatrix<float64_t> LG = layer.get_local_gradients();

	// manually compute local gradients
	SGMatrix<float64_t> A = layer.get_activations();
	SGMatrix<float64_t> LG_ref(A.num_rows, A.num_cols);
	for (int32_t i=0; i<A.num_rows*A.num_cols; i++)
		LG_ref[i] = (A[i]-y[i])/x.num_cols;

	// compare
	EXPECT_EQ(LG_ref.num_rows, LG.num_rows);
	EXPECT_EQ(LG_ref.num_cols, LG.num_cols);
	for (int32_t i=0; i<LG.num_rows*LG.num_cols; i++)
		EXPECT_NEAR(LG_ref[i], LG[i], 1e-6);
}
