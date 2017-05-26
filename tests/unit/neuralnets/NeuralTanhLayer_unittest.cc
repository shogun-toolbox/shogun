/*
 * Copyright (c) 2016, Shogun Toolbox Foundation
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
 * Written (W) 2016 Arasu Arun
 */

#include <shogun/neuralnets/NeuralTanhlayer.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

/** Compares the activations computed using the layer against manually computed
 * activations
 */
TEST(NeuralTanhlayer, compute_activations)
{
	CNeuralTanhLayer layer(9);

	// initialize some random inputs
	CMath::init_random(100);
	SGMatrix<float64_t> x(12,3);
	for (int32_t i=0; i<x.num_rows*x.num_cols; i++)
		x[i] = CMath::random(-10.0,10.0);

	CNeuralInputLayer* input = new CNeuralInputLayer (x.num_rows);
	input->set_batch_size(x.num_cols);

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(input);

	SGVector<int32_t> input_indices(1);
	input_indices[0] = 0;

	// initialize the layer
	layer.initialize_neural_layer(layers, input_indices);
	SGVector<float64_t> params(layer.get_num_parameters());
	SGVector<bool> param_regularizable(layer.get_num_parameters());
	layer.initialize_parameters(params, param_regularizable, 1.0);
	layer.set_batch_size(x.num_cols);

	// compute the layer's activations
	input->compute_activations(x);
	layer.compute_activations(params, layers);
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

			A_ref(i,j) = CMath::exp(-2.0*A_ref(i,j));
			A_ref(i,j) = (1-A_ref(i,j))/(1+A_ref(i,j));
		}
	}

	// compare
	EXPECT_EQ(A_ref.num_rows, A.num_rows);
	EXPECT_EQ(A_ref.num_cols, A.num_cols);
	for (int32_t i=0; i<A.num_rows*A.num_cols; i++)
		EXPECT_NEAR(A_ref[i], A[i], 1e-12);

	SG_UNREF(layers);
}

/** Compares the local gradients computed using the layer against gradients
 * computed using numerical approximation
 */
TEST(NeuralTanhLayer, compute_local_gradients)
{
	CNeuralTanhLayer layer(9);

	CMath::init_random(100);
	SGMatrix<float64_t> x(12,3);
	for (int32_t i=0; i<x.num_rows*x.num_cols; i++)
		x[i] = CMath::random(-10.0,10.0);

	CNeuralInputLayer* input = new CNeuralInputLayer (x.num_rows);
	input->set_batch_size(x.num_cols);

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(input);

	SGVector<int32_t> input_indices(1);
	input_indices[0] = 0;

	// initialize the layer
	layer.initialize_neural_layer(layers, input_indices);
	SGVector<float64_t> params(layer.get_num_parameters());
	SGVector<bool> param_regularizable(layer.get_num_parameters());
	layer.initialize_parameters(params, param_regularizable, 1.0);
	layer.set_batch_size(x.num_cols);

	SGMatrix<float64_t> y(layer.get_num_neurons(), x.num_cols);
	for (int32_t i=0; i<y.num_rows*y.num_cols; i++)
		y[i] = CMath::random(0.0,1.0);

	// compute the layer's local gradients
	input->compute_activations(x);
	layer.compute_activations(params, layers);
	layer.compute_local_gradients(y);
	SGMatrix<float64_t> LG = layer.get_local_gradients();

	// manually compute local gradients
	SGMatrix<float64_t> A = layer.get_activations();
	SGMatrix<float64_t> LG_numerical(A.num_rows, A.num_cols);
	float64_t epsilon = 1e-9;
	for (int32_t i=0; i<A.num_rows*A.num_cols; i++)
	{
		A[i] += epsilon;
		float64_t error_plus = layer.compute_error(y);
		A[i] -= 2*epsilon;
		float64_t error_minus = layer.compute_error(y);
		A[i] += epsilon;

		LG_numerical[i] = (error_plus-error_minus)/(2*epsilon);
		LG_numerical[i] *= 1.0-A[i]*A[i];
	}

	// compare
	EXPECT_EQ(LG_numerical.num_rows, LG.num_rows);
	EXPECT_EQ(LG_numerical.num_cols, LG.num_cols);
	for (int32_t i=0; i<LG.num_rows*LG.num_cols; i++)
		EXPECT_NEAR(LG_numerical[i], LG[i], 1e-6);

	SG_UNREF(layers);
}
