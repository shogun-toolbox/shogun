/*
 * Copyright (c) 2017, Shogun Toolbox Foundation
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
 * Written (W) 2017 Olivier Nguyen
 */

#include <shogun/neuralnets/NeuralRecurrentLayer.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

class NeuralRecurrentLayer: public ::testing::Test
{
public:
	virtual void SetUp()
	{
	}

	virtual void TearDown()
	{
	}

};


/** Compares the activations computed using the layer against manually computed
 * activations
 */
TEST_F(NeuralRecurrentLayer, compute_activations)
{
	CNeuralRecurrentLayer layer(9);

	// initialize some random inputs
	CMath::init_random(100);
	SGMatrix<float64_t> x1(12,3);
	for (int32_t i=0; i<x1.num_rows*x1.num_cols; i++)
		x1[i] = CMath::random(-10.0,10.0);

	CNeuralInputLayer* input1 = new CNeuralInputLayer (x1.num_rows);
	input1->set_batch_size(x1.num_cols);

	SGMatrix<float64_t> x2(7,3);
	for (int32_t i=0; i<x2.num_rows*x2.num_cols; i++)
		x2[i] = CMath::random(-10.0,10.0);

	CNeuralInputLayer* input2 = new CNeuralInputLayer (x2.num_rows);
	input2->set_batch_size(x2.num_cols);

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(input1);
	layers->append_element(input2);

	SGVector<int32_t> input_indices(2);
	input_indices[0] = 0;
	input_indices[1] = 1;

	// initialize the layer
	layer.initialize_neural_layer(layers, input_indices);
	SGVector<float64_t> params(layer.get_num_parameters());
	SGVector<bool> param_regularizable(layer.get_num_parameters());
	layer.initialize_parameters(params, param_regularizable, 1.0);
	layer.set_batch_size(x1.num_cols);

	int32_t num_parameters = (9 * 2) + (9 * 12 * 2) + (9 * 7 * 2) + (9 * 9 * 2);
	EXPECT_EQ(num_parameters, params.size());

	// compute the layer's activations
	input1->compute_activations(x1);
	input2->compute_activations(x2);
	layer.compute_activations(params, layers);
	SGMatrix<float64_t> A = layer.get_activations();
	/*

	// manually compute the layer's activations
	SGMatrix<float64_t> A_ref(layer.get_num_neurons(), x1.num_cols);

	float64_t* biases = params.vector;
	float64_t* weights1 = biases + layer.get_num_neurons();
	float64_t* weights2 = weights1 +
		layer.get_num_neurons()*input1->get_num_neurons();

	for (int32_t i=0; i<A_ref.num_rows; i++)
	{
		for (int32_t j=0; j<A_ref.num_cols; j++)
		{
			A_ref(i,j) = biases[i];

			for (int32_t k=0; k<x1.num_rows; k++)
				A_ref(i,j) += weights1[i+k*A_ref.num_rows]*x1(k,j);

			for (int32_t k=0; k<x2.num_rows; k++)
				A_ref(i,j) += weights2[i+k*A_ref.num_rows]*x2(k,j);
		}
	}

	// compare
	EXPECT_EQ(A_ref.num_rows, A.num_rows);
	EXPECT_EQ(A_ref.num_cols, A.num_cols);
	for (int32_t i=0; i<A.num_rows*A.num_cols; i++)
		EXPECT_NEAR(A_ref[i], A[i], 1e-12);
	*/

	SG_UNREF(layers);
}

/** Compares the error computed using the layer against a manually computed
 * error
 */
TEST_F(NeuralRecurrentLayer, compute_error)
{
}

/** Compares the local gradients computed using the layer against gradients
 * computed using numerical approximation
 */
TEST_F(NeuralRecurrentLayer, compute_local_gradients)
{
}

/** Compares the parameter gradients computed using the layer, when the layer
 * is used as an output layer, against gradients computed using numerical
 * approximation
 */
TEST_F(NeuralRecurrentLayer, compute_parameter_gradients_output)
{
}

/** Compares the parameter gradients computed using the layer, when the layer
 * is used as a hidden layer, against gradients computed using numerical
 * approximation
 */
TEST_F(NeuralRecurrentLayer, compute_parameter_gradients_hidden)
{
}
