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

#include <shogun/neuralnets/NeuralRectifiedLinearLayer.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

/** Compares the activations computed using the layer against manually computed 
 * activations
 */
TEST(NeuralRectifiedLinearLayer, compute_activations)
{
	CNeuralRectifiedLinearLayer layer(9);
	
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
	layer.initialize(layers, input_indices);
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
			
			A_ref(i,j) = CMath::max<float64_t>(0, A_ref(i,j));
		}
	}
	
	// compare
	EXPECT_EQ(A_ref.num_rows, A.num_rows);
	EXPECT_EQ(A_ref.num_cols, A.num_cols);
	for (int32_t i=0; i<A.num_rows*A.num_cols; i++)
		EXPECT_NEAR(A_ref[i], A[i], 1e-12);
	
	SG_UNREF(layers);
}

/** Compares the parameter gradients computed using the layer, when the layer 
 * is used as a hidden layer, against gradients computed using numerical
 * approximation
 */
TEST(NeuralRectifiedLinearLayer, compute_parameter_gradients_hidden)
{
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
	
	// initialize hidden the layer
	CNeuralLinearLayer* layer_hid = new CNeuralRectifiedLinearLayer(5);
	
	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(input1);
	layers->append_element(input2);
	layers->append_element(layer_hid);
	
	SGVector<int32_t> input_indices_hid(2);
	input_indices_hid[0] = 0;
	input_indices_hid[1] = 1;
	
	SGVector<int32_t> input_indices_out(1);
	input_indices_out[0] = 2;
	
	SGMatrix<float64_t> y(9,3);
	for (int32_t i=0; i<y.num_rows*y.num_cols; i++)
		y[i] = CMath::random(0.0,1.0);
	
	// initialize the hidden layer
	layer_hid->initialize(layers, input_indices_hid);
	SGVector<float64_t> param_hid(layer_hid->get_num_parameters());
	SGVector<bool> param_regularizable_hid(layer_hid->get_num_parameters());
	layer_hid->initialize_parameters(param_hid, param_regularizable_hid, 0.01);
	layer_hid->set_batch_size(x1.num_cols);
	
	// initialize the output layer
	CNeuralLinearLayer layer_out(y.num_rows);
	layer_out.initialize(layers, input_indices_out);
	SGVector<float64_t> param_out(layer_out.get_num_parameters());
	SGVector<bool> param_regularizable_out(layer_out.get_num_parameters());
	layer_out.initialize_parameters(param_out, param_regularizable_out, 0.01);
	layer_out.set_batch_size(x1.num_cols);
	
	// compute activations
	input1->compute_activations(x1);
	input2->compute_activations(x2);
	layer_hid->compute_activations(param_hid, layers);
	layer_out.compute_activations(param_out, layers);
	
	// compute gradients
	layer_hid->get_activation_gradients().zero();
	SGVector<float64_t> gradients_out(layer_out.get_num_parameters());
	layer_out.compute_gradients(param_out, y, layers, gradients_out);
	
	SGVector<float64_t> gradients_hid(layer_hid->get_num_parameters());
	layer_hid->compute_gradients(param_hid, SGMatrix<float64_t>(), 
			layers, gradients_hid);
	
	// manually compute parameter gradients
	SGVector<float64_t> gradients_hid_numerical(layer_hid->get_num_parameters());
	float64_t epsilon = 1e-9;
	for (int32_t i=0; i<layer_hid->get_num_parameters(); i++)
	{
		param_hid[i] += epsilon;
		input1->compute_activations(x1);
		input2->compute_activations(x2);
		layer_hid->compute_activations(param_hid, layers);
		layer_out.compute_activations(param_out, layers);
		float64_t error_plus = layer_out.compute_error(y);
		
		param_hid[i] -= 2*epsilon;
		input1->compute_activations(x1);
		input2->compute_activations(x2);
		layer_hid->compute_activations(param_hid, layers);
		layer_out.compute_activations(param_out, layers);
		float64_t error_minus = layer_out.compute_error(y);
		param_hid[i] += epsilon;
		
		gradients_hid_numerical[i] = (error_plus-error_minus)/(2*epsilon);
	}
	
	// compare
	for (int32_t i=0; i<gradients_hid_numerical.vlen; i++)
		EXPECT_NEAR(gradients_hid_numerical[i], gradients_hid[i], 1e-6);
	
	SG_UNREF(layers);
}
