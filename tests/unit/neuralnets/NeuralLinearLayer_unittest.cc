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
#include <gtest/gtest.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralLinearLayer.h>

using namespace shogun;

using NeuralLinearLayerTest = NeuralLayerTestFixture;

/** Compares the activations computed using the layer against manually computed
 * activations
 */
TEST_F(NeuralLinearLayerTest, compute_activations)
{
	int32_t seed = 100;
	// initialize some random inputs
	SGMatrix<float64_t> x1;
	std::shared_ptr<NeuralInputLayer> input1;
	std::mt19937_64 prng(seed);
	std::tie(x1, input1) = setup_input_layer<float64_t>(12, 3, -10.0, 10.0, prng);

	SGMatrix<float64_t> x2;
	std::shared_ptr<NeuralInputLayer> input2;
	std::tie(x2, input2) = setup_input_layer<float64_t>(7, 3, -10.0, 10.0, prng);

	// initialize the layer
	auto layer = std::make_shared<NeuralLinearLayer>(9);
	SGVector<int32_t> input_indices(2);
	input_indices[0] = 0;
	input_indices[1] = 1;
	auto params =
	    init_linear_layer(layer, input_indices, x1.num_cols, 1.0, false);

	// get the layer's activations
	SGMatrix<float64_t> A = layer->get_activations();

	// manually compute the layer's activations
	auto biases =
	    SGVector<float64_t>(params.vector, layer->get_num_neurons(), 0);
	auto weights1 = SGMatrix<float64_t>(
	    params.vector, layer->get_num_neurons(), x1.num_rows, biases.size());
	auto weights2 = SGMatrix<float64_t>(
	    params.vector, layer->get_num_neurons(), x2.num_rows,
	    static_cast<index_t>(biases.size() + weights1.size()));
	auto A_ref = shogun::linalg::add(
	    shogun::linalg::matrix_prod(weights1, x1),
	    shogun::linalg::matrix_prod(weights2, x2));
	shogun::linalg::add_vector(A_ref, biases, A_ref);

	// compare
	EXPECT_EQ(A_ref.num_rows, A.num_rows);
	EXPECT_EQ(A_ref.num_cols, A.num_cols);
	for (int32_t i=0; i<A.num_rows*A.num_cols; i++)
		EXPECT_NEAR(A_ref[i], A[i], 1e-12);
}

/** Compares the error computed using the layer against a manually computed
 * error
 */
TEST_F(NeuralLinearLayerTest, compute_error)
{
	int32_t seed = 100;
	// initialize some random inputs
	SGMatrix<float64_t> x1;
	std::mt19937_64 prng(seed);
	std::shared_ptr<NeuralInputLayer> input1;
	std::tie(x1, input1) = setup_input_layer<float64_t>(12, 3, -10.0, 10.0, prng);

	SGMatrix<float64_t> x2;
	std::shared_ptr<NeuralInputLayer> input2;
	std::tie(x2, input2) = setup_input_layer<float64_t>(7, 3, -10.0, 10.0, prng);

	// initialize the layer
	auto layer=std::make_shared<NeuralLinearLayer>(9);
	SGVector<int32_t> input_indices(2);
	input_indices[0] = 0;
	input_indices[1] = 1;
	auto params =
	    init_linear_layer(layer, input_indices, x1.num_cols, 1.0, false);

	// initialize output
	auto y = create_rand_matrix<float64_t>(
	    layer->get_num_neurons(), x1.num_cols, 0.0, 1.0, prng);

	// get the layer's activations and error
	SGMatrix<float64_t> A = layer->get_activations();
	float64_t error = layer->compute_error(y);

	// manually compute error
	float64_t error_ref = 0;
	for (int32_t i=0; i<A.num_rows*A.num_cols; i++)
		error_ref += 0.5*Math::pow(y[i]-A[i],2)/y.num_cols;

	// compare
	EXPECT_NEAR(error_ref, error, 1e-11);
}

/** Compares the local gradients computed using the layer against gradients
 * computed using numerical approximation
 */
TEST_F(NeuralLinearLayerTest, compute_local_gradients)
{
	int32_t seed = 100;
	// initialize some random inputs
	SGMatrix<float64_t> x;
	std::shared_ptr<NeuralInputLayer> input;
	std::mt19937_64 prng(seed);
	std::tie(x, input) = setup_input_layer<float64_t>(12, 3, -10.0, 10.0, prng);

	// initialize the layer
	auto layer=std::make_shared<NeuralLinearLayer>(9);
	SGVector<int32_t> input_indices(1);
	input_indices[0] = 0;
	auto params =
	    init_linear_layer(layer, input_indices, x.num_cols, 0.01, false);

	// initialize output
	auto y = create_rand_matrix<float64_t>(
	    layer->get_num_neurons(), x.num_cols, 0.0, 1.0, prng);

	// compute the layer's local gradients
	layer->compute_local_gradients(y);
	SGMatrix<float64_t> LG = layer->get_local_gradients();

	// manually compute local gradients
	SGMatrix<float64_t> A = layer->get_activations();
	SGMatrix<float64_t> LG_numerical(A.num_rows, A.num_cols);
	float64_t epsilon = 1e-9;
	for (int32_t i=0; i<A.num_rows*A.num_cols; i++)
	{
		A[i] += epsilon;
		float64_t error_plus = layer->compute_error(y);
		A[i] -= 2*epsilon;
		float64_t error_minus = layer->compute_error(y);
		A[i] += epsilon;

		LG_numerical[i] = (error_plus-error_minus)/(2*epsilon);
	}

	// compare
	EXPECT_EQ(LG_numerical.num_rows, LG.num_rows);
	EXPECT_EQ(LG_numerical.num_cols, LG.num_cols);
	for (int32_t i = 0; i < LG.num_rows * LG.num_cols; i++)
	{
		EXPECT_NEAR(LG_numerical[i], LG[i], 1e-6);
	}
}

/** Compares the parameter gradients computed using the layer, when the layer
 * is used as an output layer, against gradients computed using numerical
 * approximation
 */
TEST_F(NeuralLinearLayerTest, compute_parameter_gradients_output)
{
	int32_t seed = 100;
	// initialize some random inputs
	SGMatrix<float64_t> x1;
	std::shared_ptr<NeuralInputLayer> input1;
	std::mt19937_64 prng(seed);
	std::tie(x1, input1) = setup_input_layer<float64_t>(12, 3, -10.0, 10.0, prng);

	SGMatrix<float64_t> x2;
	std::shared_ptr<NeuralInputLayer> input2;
	std::tie(x2, input2) = setup_input_layer<float64_t>(7, 3, -10.0, 10.0, prng);

	// initialize the layer
	auto layer=std::make_shared<NeuralLinearLayer>(9);
	SGVector<int32_t> input_indices(2);
	input_indices[0] = 0;
	input_indices[1] = 1;
	auto params =
	    init_linear_layer(layer, input_indices, x1.num_cols, 0.01, false);

	// initialize output
	auto y = create_rand_matrix<float64_t>(
	    layer->get_num_neurons(), x1.num_cols, 0.0, 1.0, prng);

	// compute parameter gradients
	SGVector<float64_t> gradients(layer->get_num_parameters());
	layer->compute_gradients(params, y, m_layers, gradients);

	// manually compute parameter gradients
	SGVector<float64_t> gradients_numerical(layer->get_num_parameters());
	float64_t epsilon = 1e-9;
	for (int32_t i=0; i<layer->get_num_parameters(); i++)
	{
		params[i] += epsilon;
		input1->compute_activations(x1);
		input2->compute_activations(x2);
		layer->compute_activations(params, m_layers);
		float64_t error_plus = layer->compute_error(y);

		params[i] -= 2*epsilon;
		input1->compute_activations(x1);
		input2->compute_activations(x2);
		layer->compute_activations(params, m_layers);
		float64_t error_minus = layer->compute_error(y);
		params[i] += epsilon;

		gradients_numerical[i] = (error_plus-error_minus)/(2*epsilon);
	}

	// compare
	for (int32_t i = 0; i < gradients.vlen; i++)
	{
		EXPECT_NEAR(gradients_numerical[i], gradients[i], 1e-6);
	}
}

/** Compares the parameter gradients computed using the layer, when the layer
 * is used as a hidden layer, against gradients computed using numerical
 * approximation
 */
TEST_F(NeuralLinearLayerTest, compute_parameter_gradients_hidden)
{
	int32_t seed = 100;
	// initialize some random inputs
	SGMatrix<float64_t> x1;
	std::shared_ptr<NeuralInputLayer> input1;
	std::mt19937_64 prng(seed);
	std::tie(x1, input1) = setup_input_layer<float64_t>(12, 3, -10.0, 10.0, prng);

	SGMatrix<float64_t> x2;
	std::shared_ptr<NeuralInputLayer> input2;
	std::tie(x2, input2) = setup_input_layer<float64_t>(7, 3, -10.0, 10.0, prng);

	// initialize the hidden layer
	auto layer_hid = std::make_shared<NeuralLinearLayer>(5);
	SGVector<int32_t> input_indices_hid(2);
	input_indices_hid[0] = 0;
	input_indices_hid[1] = 1;
	auto params_hid = init_linear_layer(
	    layer_hid, input_indices_hid, x1.num_cols, 0.01, true);

	// initialize the output layer
	auto layer_out=std::make_shared<NeuralLinearLayer>(9);
	SGVector<int32_t> input_indices_out(1);
	input_indices_out[0] = 2;
	auto params_out = init_linear_layer(
	    layer_out, input_indices_out, x1.num_cols, 0.01, false);

	// initialize output
	auto y = create_rand_matrix<float64_t>(
	    layer_out->get_num_neurons(), x1.num_cols, 0.0, 1.0, prng);

	// compute gradients
	layer_hid->get_activation_gradients().zero();
	SGVector<float64_t> gradients_out(layer_out->get_num_parameters());
	layer_out->compute_gradients(params_out, y, m_layers, gradients_out);

	SGVector<float64_t> gradients_hid(layer_hid->get_num_parameters());
	layer_hid->compute_gradients(
	    params_hid, SGMatrix<float64_t>(), m_layers, gradients_hid);

	// manually compute parameter gradients
	SGVector<float64_t> gradients_hid_numerical(layer_hid->get_num_parameters());
	float64_t epsilon = 1e-9;
	for (int32_t i=0; i<layer_hid->get_num_parameters(); i++)
	{
		params_hid[i] += epsilon;
		input1->compute_activations(x1);
		input2->compute_activations(x2);
		layer_hid->compute_activations(params_hid, m_layers);
		layer_out->compute_activations(params_out, m_layers);
		float64_t error_plus = layer_out->compute_error(y);

		params_hid[i] -= 2 * epsilon;
		input1->compute_activations(x1);
		input2->compute_activations(x2);
		layer_hid->compute_activations(params_hid, m_layers);
		layer_out->compute_activations(params_out, m_layers);
		float64_t error_minus = layer_out->compute_error(y);
		params_hid[i] += epsilon;

		gradients_hid_numerical[i] = (error_plus-error_minus)/(2*epsilon);
	}

	// compare
	for (int32_t i = 0; i < gradients_hid_numerical.vlen; i++)
	{
		EXPECT_NEAR(gradients_hid_numerical[i], gradients_hid[i], 1e-6);
	}
}
