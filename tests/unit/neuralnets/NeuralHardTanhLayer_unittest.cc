/*
 * Copyright (c) 2016 Shogun Toolbox Foundation
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

 * Written (W) 2016 Arasu Arun
 */

#include <shogun/neuralnets/NeuralHardTanhLayer.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

/** Compares the activations computed using the layer against manually computed
 * activations
 */
TEST(NeuralHardTanhLayer, compute_activations)
{
	CNeuralHardTanhLayer layer(9);
	float64_t min_act = -1.0;
	float64_t max_act = 1.0;
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
	layer.set_min_act(min_act);
	layer.set_max_act(max_act);
	layer.compute_activations(params, layers);
	SGMatrix<float64_t> A = layer.get_activations();

	// manually compute the layer's activations
	SGMatrix<float64_t> A_ref(layer.get_num_neurons(), x.num_cols);

	SGVector<float64_t> biases = params;
	SGVector<float64_t> weights(params.vector,params.vlen,layer.get_num_neurons());

	for (int32_t i=0; i<A_ref.num_rows; i++)
	{
		for (int32_t j=0; j<A_ref.num_cols; j++)
		{
			A_ref(i,j) = biases[i];

			for (int32_t k=0; k<x.num_rows; k++)
				A_ref(i,j) += weights[i+k*A_ref.num_rows]*x(k,j);

			A_ref(i,j) = A_ref(i,j) >= max_act ? max_act : 
				CMath::max<float64_t>(A_ref(i,j),min_act);
		}
	}

	// compare
	EXPECT_EQ(A_ref.num_rows, A.num_rows);
	EXPECT_EQ(A_ref.num_cols, A.num_cols);
	for (int32_t i=0; i<A.num_rows*A.num_cols; i++)
		EXPECT_NEAR(A_ref[i], A[i], 1e-12);

	SG_UNREF(layers);
}
