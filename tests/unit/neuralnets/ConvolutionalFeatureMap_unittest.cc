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

#include <shogun/neuralnets/ConvolutionalFeatureMap.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralLinearLayer.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(ConvolutionalFeatureMap, compute_activations)
{
	const int32_t w = 6;
	const int32_t h = 5;
	const int32_t rx = 1;
	const int32_t ry = 1;
	const int32_t b = 2;
	const int32_t map_index = 1;
	const int32_t num_maps = 3;
	
	SGMatrix<float64_t> x1(w*h,b);
	for (int32_t i=0; i<x1.num_rows*x1.num_cols; i++)
		x1[i] = i;
	
	CNeuralInputLayer* input1 = new CNeuralInputLayer (x1.num_rows);
	input1->set_batch_size(x1.num_cols);
	
	// two channels
	SGMatrix<float64_t> x2(2*w*h,b);
	for (int32_t i=0; i<x2.num_rows*x2.num_cols; i++)
		x2[i] = float64_t(i)/8;
	
	CNeuralInputLayer* input2 = new CNeuralInputLayer (x2.num_rows);
	input2->set_batch_size(x2.num_cols);
	
	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(input1);
	layers->append_element(input2);
	
	SGVector<int32_t> input_indices(2);
	input_indices[0] = 0;
	input_indices[1] = 1;
	
	CConvolutionalFeatureMap map(w,h,rx,ry,map_index);
	SGVector<float64_t> params(w*h+(2*rx+1)*(2*ry+1));
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = float64_t(i)/4;
	
	input1->compute_activations(x1);
	input2->compute_activations(x2);
	
	SGMatrix<float64_t> A(num_maps*w*h,b);
	SGMatrix<float64_t> buffer(num_maps*w*h,b);
	map.compute_activations(params, layers, input_indices, A, buffer);
	
	// reference numbers generated using scipy.signal.convolve2d
	float64_t ref[] = { 
		235.00000, 388.06250, 449.25000, 510.43750, 367.25000, 515.93750, 
		832.12500, 928.00000,1023.87500, 726.93750, 831.25000,1311.50000,
		1407.37500,1503.25000,1051.62500,1146.56250,1790.87500,1886.75000,
		1982.62500,1376.31250,1461.87500,2270.25000,2366.12500,2462.00000,
		1701.00000,1138.75000,1762.12500,1828.93750,1895.75000,1307.25000,
		1675.00000,2581.81250,2643.00000,2704.18750,1852.25000,2777.18750,
		4274.62500,4370.50000,4466.37500,3055.68750,3092.50000,4754.00000,
		4849.87500,4945.75000,3380.37500,3407.81250,5233.37500,5329.25000,
		5425.12500,3705.06250,3723.12500,5712.75000,5808.62500,5904.50000,
		4029.75000,2713.75000,4158.37500,4225.18750,4292.00000,2927.25000 };
	
	for (int32_t i=0; i<w*h; i++)
		for (int32_t j=0; j<A.num_cols; j++)
			EXPECT_NEAR(ref[i+j*w*h], A(i+map_index*w*h,j), 1.0e-15);
			
	SG_UNREF(layers);
}

TEST(ConvolutionalFeatureMap, compute_activations_logistic)
{
	const int32_t w = 6;
	const int32_t h = 5;
	const int32_t rx = 1;
	const int32_t ry = 1;
	const int32_t b = 2;
	const int32_t map_index = 1;
	const int32_t num_maps = 3;
	
	SGMatrix<float64_t> x1(w*h,b);
	for (int32_t i=0; i<x1.num_rows*x1.num_cols; i++)
		x1[i] = i;
	
	CNeuralInputLayer* input1 = new CNeuralInputLayer (x1.num_rows);
	input1->set_batch_size(x1.num_cols);
	
	// two channels
	SGMatrix<float64_t> x2(2*w*h,b);
	for (int32_t i=0; i<x2.num_rows*x2.num_cols; i++)
		x2[i] = float64_t(i)/8;
	
	CNeuralInputLayer* input2 = new CNeuralInputLayer (x2.num_rows);
	input2->set_batch_size(x2.num_cols);
	
	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(input1);
	layers->append_element(input2);
	
	SGVector<int32_t> input_indices(2);
	input_indices[0] = 0;
	input_indices[1] = 1;
	
	CConvolutionalFeatureMap map(w,h,rx,ry,map_index,CMAF_LOGISTIC);
	SGVector<float64_t> params(w*h+(2*rx+1)*(2*ry+1));
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = float64_t(i)*1e-4;
	
	input1->compute_activations(x1);
	input2->compute_activations(x2);
	
	SGMatrix<float64_t> A(num_maps*w*h,b);
	SGMatrix<float64_t> buffer(num_maps*w*h,b);
	map.compute_activations(params, layers, input_indices, A, buffer);
	
	float64_t ref[] = { 
		   0.52348,   0.53873,   0.54480,   0.55087,   0.53666,   0.55141,
		   0.58245,   0.59175,   0.60098,   0.57219,   0.58237,   0.62822,
		   0.63713,   0.64595,   0.60364,   0.61269,   0.67180,   0.68020,
		   0.68849,   0.63426,   0.64216,   0.71261,   0.72040,   0.72806,
		   0.66383,   0.61195,   0.66926,   0.67515,   0.68098,   0.62783,
		   0.66150,   0.73744,   0.74216,   0.74681,   0.67719,   0.75229,
		   0.84682,   0.85173,   0.85650,   0.77246,   0.77504,   0.87007,
		   0.87435,   0.87850,   0.79448,   0.79627,   0.89026,   0.89395,
		   0.89753,   0.81488,   0.81597,   0.90764,   0.91080,   0.91387,
		   0.83368,   0.74753,   0.84069,   0.84423,   0.84772,   0.76331 };
	
	for (int32_t i=0; i<w*h; i++)
		for (int32_t j=0; j<A.num_cols; j++)
			EXPECT_NEAR(ref[i+j*w*h], A(i+map_index*w*h,j), 1.0e-5);
			
	SG_UNREF(layers);
}

TEST(ConvolutionalFeatureMap, compute_activations_rectified_linear)
{
	const int32_t w = 6;
	const int32_t h = 5;
	const int32_t rx = 1;
	const int32_t ry = 1;
	const int32_t b = 2;
	const int32_t map_index = 1;
	const int32_t num_maps = 3;
	
	SGMatrix<float64_t> x1(w*h,b);
	for (int32_t i=0; i<x1.num_rows*x1.num_cols; i++)
		x1[i] = i;
	
	CNeuralInputLayer* input1 = new CNeuralInputLayer (x1.num_rows);
	input1->set_batch_size(x1.num_cols);
	
	// two channels
	SGMatrix<float64_t> x2(2*w*h,b);
	for (int32_t i=0; i<x2.num_rows*x2.num_cols; i++)
		x2[i] = float64_t(i)/8;
	
	CNeuralInputLayer* input2 = new CNeuralInputLayer (x2.num_rows);
	input2->set_batch_size(x2.num_cols);
	
	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(input1);
	layers->append_element(input2);
	
	SGVector<int32_t> input_indices(2);
	input_indices[0] = 0;
	input_indices[1] = 1;
	
	CConvolutionalFeatureMap map(w,h,rx,ry,map_index,CMAF_RECTIFIED_LINEAR);
	SGVector<float64_t> params(w*h+(2*rx+1)*(2*ry+1));
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = float64_t(i)/4 - 8;
	
	input1->compute_activations(x1);
	input2->compute_activations(x2);
	
	SGMatrix<float64_t> A(num_maps*w*h,b);
	SGMatrix<float64_t> buffer(num_maps*w*h,b);
	map.compute_activations(params, layers, input_indices, A, buffer);
	
	float64_t ref[] = { 
		      0.00000,   0.00000,   0.00000,   0.00000,   0.00000,   0.00000,
			  14.12500,  20.00000,  25.87500,  28.93750,  13.25000,  43.50000,
			  49.37500,  55.25000,  53.62500,  28.56250,  72.87500,  78.75000,
			  84.62500,  78.31250,  43.87500, 102.25000, 108.12500, 114.00000,
			  103.00000,  90.75000, 164.12500, 170.93750, 177.75000, 139.25000,
			  0.00000,  23.81250,  25.00000,  26.18750,  44.25000,  99.18750,
			  216.62500, 222.50000, 228.37500, 197.68750, 114.50000, 246.00000,
			  251.87500, 257.75000, 222.37500, 129.81250, 275.37500, 281.25000,
			  287.12500, 247.06250, 145.12500, 304.75000, 310.62500, 316.50000,
			  271.75000, 225.75000, 400.37500, 407.18750, 414.00000, 319.25000 };
	
	for (int32_t i=0; i<w*h; i++)
		for (int32_t j=0; j<A.num_cols; j++)
			EXPECT_NEAR(ref[i+j*w*h], A(i+map_index*w*h,j), 1.0e-15);
			
	SG_UNREF(layers);
}

TEST(ConvolutionalFeatureMap, compute_parameter_gradients)
{
	const int32_t w = 6;
	const int32_t h = 5;
	const int32_t rx = 1;
	const int32_t ry = 1;
	const int32_t b = 2;
	const int32_t map_index = 1;
	const int32_t num_maps = 3;
	
	CMath::init_random(10);
	
	SGMatrix<float64_t> x1(w*h,b);
	for (int32_t i=0; i<x1.num_rows*x1.num_cols; i++)
		x1[i] = CMath::random(-10.0,10.0);
	
	CNeuralInputLayer* input1 = new CNeuralInputLayer (x1.num_rows);
	input1->set_batch_size(x1.num_cols);
	
	// two channels
	SGMatrix<float64_t> x2(2*w*h,b);
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
	
	CConvolutionalFeatureMap map(w,h,rx,ry,map_index);
	SGVector<float64_t> params(w*h+(2*rx+1)*(2*ry+1));
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = CMath::normal_random(0.0,0.01);
	
	input1->compute_activations(x1);
	input2->compute_activations(x2);
	
	SGMatrix<float64_t> A(num_maps*w*h,b);
	SGMatrix<float64_t> buffer(num_maps*w*h,b);
	A.zero();
	
	map.compute_activations(params, layers, input_indices, A, buffer);
	
	// compute activation gradients with respect to some function
	// assuming the function is 0.5*sum(A[i]^2)
	SGMatrix<float64_t> AG(num_maps*w*h,b);
	for (int32_t i=0; i<AG.num_rows*AG.num_cols; i++)
		AG[i] = A[i];
	
	// compute parameter gradients
	SGVector<float64_t> PG(params.vlen);
	map.compute_gradients(params, A, AG, layers, input_indices, PG);
	
	// approximate parameter gradients
	SGVector<float64_t> PG_numerical(params.vlen);
	float64_t epsilon = 1e-9;
	for (int32_t i=0; i<params.vlen; i++)
	{
		params[i] += epsilon;
		map.compute_activations(params, layers, input_indices, A, buffer);
		float64_t error_plus = 0;
		for (int32_t k=0; k<A.num_rows*A.num_cols; k++)
			error_plus += 0.5*A[k]*A[k];
		
		params[i] -= 2*epsilon;
		map.compute_activations(params, layers, input_indices, A, buffer);
		float64_t error_minus = 0;
		for (int32_t k=0; k<A.num_rows*A.num_cols; k++)
			error_minus += 0.5*A[k]*A[k];
		
		params[i] += epsilon;
		
		PG_numerical[i] = (error_plus-error_minus)/(2*epsilon);
	}
	
	// compare
	for (int32_t i=0; i<PG.vlen; i++)
		EXPECT_NEAR(PG_numerical[i], PG[i], 1e-5);
	
	SG_UNREF(layers);
}

TEST(ConvolutionalFeatureMap, compute_parameter_gradients_logistic)
{
	const int32_t w = 6;
	const int32_t h = 5;
	const int32_t rx = 1;
	const int32_t ry = 1;
	const int32_t b = 2;
	
	CMath::init_random(10);
	
	SGMatrix<float64_t> x1(w*h,b);
	for (int32_t i=0; i<x1.num_rows*x1.num_cols; i++)
		x1[i] = CMath::random(-10.0,10.0);
	
	CNeuralInputLayer* input1 = new CNeuralInputLayer (x1.num_rows);
	input1->set_batch_size(x1.num_cols);

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(input1);
	
	SGVector<int32_t> input_indices(1);
	input_indices[0] = 0;
	
	CConvolutionalFeatureMap map(w,h,rx,ry,0, CMAF_LOGISTIC);
	SGVector<float64_t> params(w*h+(2*rx+1)*(2*ry+1));
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = CMath::normal_random(0.0,0.01);
	
	input1->compute_activations(x1);
	
	SGMatrix<float64_t> A(w*h,b);
	SGMatrix<float64_t> buffer(w*h,b);
	A.zero();
	
	map.compute_activations(params, layers, input_indices, A, buffer);
	
	// compute activation gradients with respect to some function
	// assuming the function is 0.5*sum(A[i]^2)
	SGMatrix<float64_t> AG(w*h,b);
	for (int32_t i=0; i<AG.num_rows*AG.num_cols; i++)
		AG[i] = A[i];
	
	// compute parameter gradients
	SGVector<float64_t> PG(params.vlen);
	map.compute_gradients(params, A, AG, layers, input_indices, PG);
	
	// approximate parameter gradients
	SGVector<float64_t> PG_numerical(params.vlen);
	float64_t epsilon = 1e-9;
	for (int32_t i=0; i<params.vlen; i++)
	{
		params[i] += epsilon;
		map.compute_activations(params, layers, input_indices, A, buffer);
		float64_t error_plus = 0;
		for (int32_t k=0; k<A.num_rows*A.num_cols; k++)
			error_plus += 0.5*A[k]*A[k];
		
		params[i] -= 2*epsilon;
		map.compute_activations(params, layers, input_indices, A, buffer);
		float64_t error_minus = 0;
		for (int32_t k=0; k<A.num_rows*A.num_cols; k++)
			error_minus += 0.5*A[k]*A[k];
		
		params[i] += epsilon;
		
		PG_numerical[i] = (error_plus-error_minus)/(2*epsilon);
	}
	
	// compare
	for (int32_t i=0; i<PG.vlen; i++)
		EXPECT_NEAR(PG_numerical[i], PG[i], 1e-5);
	
	SG_UNREF(layers);
}

TEST(ConvolutionalFeatureMap, compute_parameter_gradients_rectified_linear)
{
	const int32_t w = 6;
	const int32_t h = 5;
	const int32_t rx = 1;
	const int32_t ry = 1;
	const int32_t b = 2;
	
	CMath::init_random(10);
	
	SGMatrix<float64_t> x1(w*h,b);
	for (int32_t i=0; i<x1.num_rows*x1.num_cols; i++)
		x1[i] = CMath::random(-10.0,10.0);
	
	CNeuralInputLayer* input1 = new CNeuralInputLayer (x1.num_rows);
	input1->set_batch_size(x1.num_cols);

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(input1);
	
	SGVector<int32_t> input_indices(1);
	input_indices[0] = 0;
	
	CConvolutionalFeatureMap map(w,h,rx,ry,0, CMAF_RECTIFIED_LINEAR);
	SGVector<float64_t> params(w*h+(2*rx+1)*(2*ry+1));
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = CMath::normal_random(0.0,0.01);
	
	input1->compute_activations(x1);
	
	SGMatrix<float64_t> A(w*h,b);
	SGMatrix<float64_t> buffer(w*h,b);
	A.zero();
	
	map.compute_activations(params, layers, input_indices, A, buffer);
	
	// compute activation gradients with respect to some function
	// assuming the function is 0.5*sum(A[i]^2)
	SGMatrix<float64_t> AG(w*h,b);
	for (int32_t i=0; i<AG.num_rows*AG.num_cols; i++)
		AG[i] = A[i];
	
	// compute parameter gradients
	SGVector<float64_t> PG(params.vlen);
	map.compute_gradients(params, A, AG, layers, input_indices, PG);
	
	// approximate parameter gradients
	SGVector<float64_t> PG_numerical(params.vlen);
	float64_t epsilon = 1e-9;
	for (int32_t i=0; i<params.vlen; i++)
	{
		params[i] += epsilon;
		map.compute_activations(params, layers, input_indices, A, buffer);
		float64_t error_plus = 0;
		for (int32_t k=0; k<A.num_rows*A.num_cols; k++)
			error_plus += 0.5*A[k]*A[k];
		
		params[i] -= 2*epsilon;
		map.compute_activations(params, layers, input_indices, A, buffer);
		float64_t error_minus = 0;
		for (int32_t k=0; k<A.num_rows*A.num_cols; k++)
			error_minus += 0.5*A[k]*A[k];
		
		params[i] += epsilon;
		
		PG_numerical[i] = (error_plus-error_minus)/(2*epsilon);
	}
	
	// compare
	for (int32_t i=0; i<PG.vlen; i++)
		EXPECT_NEAR(PG_numerical[i], PG[i], 1e-5);
	
	SG_UNREF(layers);
}

TEST(ConvolutionalFeatureMap, compute_input_gradients)
{
	const int32_t w = 6;
	const int32_t h = 5;
	const int32_t rx = 1;
	const int32_t ry = 1;
	const int32_t b = 2;
	const int32_t map_index = 0;
	const int32_t num_maps = 1;
	
	CMath::init_random(100);
	
	CNeuralLinearLayer* input1 = new CNeuralLinearLayer (w*h);
	input1->set_batch_size(b);
	
	// two channels
	CNeuralLinearLayer* input2 = new CNeuralLinearLayer (2*w*h);
	input2->set_batch_size(b);
	
	for (int32_t i=0; i<input1->get_num_neurons()*b; i++)
		input1->get_activations()[i] = CMath::random(-10.0,10.0);
	
	for (int32_t i=0; i<input2->get_num_neurons()*b; i++)
		input2->get_activations()[i] = CMath::random(-10.0,10.0);
	
	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(input1);
	layers->append_element(input2);
	
	SGVector<int32_t> input_indices(2);
	input_indices[0] = 0;
	input_indices[1] = 1;
	
	CConvolutionalFeatureMap map(w,h,rx,ry,map_index);
	SGVector<float64_t> params(w*h+(2*rx+1)*(2*ry+1));
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = CMath::normal_random(0.0,0.01);
	
	SGMatrix<float64_t> A(num_maps*w*h,b);
	SGMatrix<float64_t> buffer(num_maps*w*h,b);
	A.zero();
	
	map.compute_activations(params, layers, input_indices, A, buffer);
	
	// compute activation gradients with respect to some function
	// assuming the function is 0.5*sum(A[i]^2)
	SGMatrix<float64_t> AG(num_maps*w*h,b);
	for (int32_t i=0; i<AG.num_rows*AG.num_cols; i++)
		AG[i] = A[i];
	
	// compute gradients
	input1->get_activation_gradients().zero();
	input2->get_activation_gradients().zero();
	SGVector<float64_t> PG(params.vlen);
	map.compute_gradients(params, A, AG, layers, input_indices, PG);
	
	// approximate input gradients
	float64_t epsilon = 1e-9;
	
	SGMatrix<float64_t> IG1(input1->get_num_neurons(), b);
	for (int32_t i=0; i<IG1.num_rows*IG1.num_cols; i++)
	{
		input1->get_activations()[i] += epsilon;
		map.compute_activations(params, layers, input_indices, A, buffer);
		float64_t error_plus = 0;
		for (int32_t k=0; k<A.num_rows*A.num_cols; k++)
			error_plus += 0.5*A[k]*A[k];
		
		input1->get_activations()[i] -= 2*epsilon;
		map.compute_activations(params, layers, input_indices, A, buffer);
		float64_t error_minus = 0;
		for (int32_t k=0; k<A.num_rows*A.num_cols; k++)
			error_minus += 0.5*A[k]*A[k];
		
		input1->get_activations()[i] += epsilon;
		
		IG1[i] = (error_plus-error_minus)/(2*epsilon);
	}
	
	SGMatrix<float64_t> IG2(input2->get_num_neurons(), b);
	for (int32_t i=0; i<IG2.num_rows*IG2.num_cols; i++)
	{
		input2->get_activations()[i] += epsilon;
		map.compute_activations(params, layers, input_indices, A, buffer);
		float64_t error_plus = 0;
		for (int32_t k=0; k<A.num_rows*A.num_cols; k++)
			error_plus += 0.5*A[k]*A[k];
		
		input2->get_activations()[i] -= 2*epsilon;
		map.compute_activations(params, layers, input_indices, A, buffer);
		float64_t error_minus = 0;
		for (int32_t k=0; k<A.num_rows*A.num_cols; k++)
			error_minus += 0.5*A[k]*A[k];
		
		input2->get_activations()[i] += epsilon;
		
		IG2[i] = (error_plus-error_minus)/(2*epsilon);
	}
	
	// compare
	for (int32_t i=0; i<IG1.num_rows*IG1.num_cols; i++)
		EXPECT_NEAR(IG1[i], input1->get_activation_gradients()[i], 1e-5);
	
	for (int32_t i=0; i<IG2.num_rows*IG2.num_cols; i++)
		EXPECT_NEAR(IG2[i], input2->get_activation_gradients()[i], 1e-5);
	
	SG_UNREF(layers);
}

