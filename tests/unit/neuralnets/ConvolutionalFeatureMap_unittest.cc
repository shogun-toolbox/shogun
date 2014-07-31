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
	
	CConvolutionalFeatureMap map(w,h,rx,ry,1,1,map_index);
	SGVector<float64_t> params(1+(2*rx+1)*(2*ry+1));
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = float64_t(i)/4;
	
	input1->compute_activations(x1);
	input2->compute_activations(x2);
	
	SGMatrix<float64_t> A(num_maps*w*h,b);
	SGMatrix<float64_t> buffer(w*h,b);
	map.compute_activations(params, layers, input_indices, A, buffer);
	
	// reference numbers generated using scipy.signal.convolve2d
	float64_t ref[] = { 
		  17.50000,  34.37500,  40.93750,  47.50000,  40.00000,  52.50000,  
		  96.56250, 110.62500, 124.68750,  99.37500,  94.68750, 166.87500, 
		  180.93750, 195.00000, 150.93750, 136.87500, 237.18750, 251.25000, 
		  265.31250, 202.50000, 179.06250, 307.50000, 321.56250, 335.62500, 
		  254.06250, 190.00000, 314.68750, 326.87500, 339.06250, 248.75000, 
		  152.50000, 270.62500, 277.18750, 283.75000, 220.00000, 356.25000, 
		  602.81250, 616.87500, 630.93750, 470.62500, 398.43750, 673.12500,
		  687.18750, 701.25000, 522.18750, 440.62500, 743.43750, 757.50000, 
		  771.56250, 573.75000, 482.81250, 813.75000, 827.81250, 841.87500, 
		  625.31250, 460.00000, 753.43750, 765.62500, 777.81250, 563.75000};
	
	for (int32_t i=0; i<w*h; i++)
		for (int32_t j=0; j<A.num_cols; j++)
			EXPECT_NEAR(ref[i+j*w*h], A(i+map_index*w*h,j), 1.0e-15);
			
	SG_UNREF(layers);
}

TEST(ConvolutionalFeatureMap, compute_activations_with_stride)
{
	const int32_t w = 12;
	const int32_t h = 10;
	const int32_t rx = 1;
	const int32_t ry = 1;
	const int32_t b = 2;
	const int32_t map_index = 1;
	const int32_t num_maps = 3;
	
	int32_t stride_x = 3;
	int32_t stride_y = 2;
	int32_t w_out = w/stride_x;
	int32_t h_out = h/stride_y;
	
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
	
	CConvolutionalFeatureMap map(w,h,rx,ry,stride_x,stride_y,map_index);
	SGVector<float64_t> params(1+(2*rx+1)*(2*ry+1));
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = float64_t(i)/4;
	
	input1->compute_activations(x1);
	input2->compute_activations(x2);
	
	SGMatrix<float64_t> A(num_maps*w_out*h_out,b);
	SGMatrix<float64_t> buffer(w*h,b);
	map.compute_activations(params, layers, input_indices, A, buffer);
	
	// reference numbers generated using scipy.signal.convolve2d
	float64_t ref[] = { 
		  55.93750, 109.37500, 122.50000, 135.62500, 148.75000, 320.62500, 
		  560.62500, 588.75000, 616.87500, 645.00000, 573.75000, 982.50000,
		  1010.62500,1038.75000,1066.87500, 826.87500,1404.37500,1432.50000,
		  1460.62500,1488.75000, 595.93750,1054.37500,1067.50000,1080.62500,
		  1093.75000,1535.62500,2585.62500,2613.75000,2641.87500,2670.00000,
		  1788.75000,3007.50000,3035.62500,3063.75000,3091.87500,2041.87500,
		  3429.37500,3457.50000,3485.62500,3513.75000};

	for (int32_t i=0; i<w_out*h_out; i++)
		for (int32_t j=0; j<A.num_cols; j++)
			EXPECT_NEAR(ref[i+j*w_out*h_out], A(i+map_index*w_out*h_out,j), 1.0e-15);
			
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
	
	CConvolutionalFeatureMap map(w,h,rx,ry,1,1,map_index,CMAF_LOGISTIC);
	SGVector<float64_t> params(1+(2*rx+1)*(2*ry+1));
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = float64_t(i)*1e-4;
	
	input1->compute_activations(x1);
	input2->compute_activations(x2);
	
	SGMatrix<float64_t> A(num_maps*w*h,b);
	SGMatrix<float64_t> buffer(w*h,b);
	map.compute_activations(params, layers, input_indices, A, buffer);
	
	float64_t ref[] = { 
		0.50175,   0.50344,   0.50409,   0.50475,   0.50400,   0.50525,   
		0.50966,   0.51106,   0.51247,   0.50994,   0.50947,   0.51668,   
		0.51809,   0.51949,   0.51509,   0.51368,   0.52370,   0.52510,   
		0.52651,   0.52024,   0.51790,   0.53071,   0.53211,   0.53351,   
		0.52538,   0.51899,   0.53143,   0.53264,   0.53385,   0.52485,   
		0.51525,   0.52704,   0.52769,   0.52834,   0.52199,   0.53556,   
		0.55999,   0.56138,   0.56276,   0.54692,   0.53976,   0.56691,   
		0.56829,   0.56967,   0.55203,   0.54395,   0.57380,   0.57518,   
		0.57655,   0.55712,   0.54813,   0.58066,   0.58203,   0.58340,   
		0.56221,   0.54587,   0.57478,   0.57597,   0.57716,   0.55614};
	
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
	
	CConvolutionalFeatureMap map(w,h,rx,ry,1,1,map_index,CMAF_RECTIFIED_LINEAR);
	SGVector<float64_t> params(1+(2*rx+1)*(2*ry+1));
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = float64_t(i)/4 - 1;
	
	input1->compute_activations(x1);
	input2->compute_activations(x2);
	
	SGMatrix<float64_t> A(num_maps*w*h,b);
	SGMatrix<float64_t> buffer(w*h,b);
	map.compute_activations(params, layers, input_indices, A, buffer);
	
	float64_t ref[] = { 
		-0.00000,  -0.00000,  -0.00000,  -0.00000,  -0.00000,  -0.00000,  
		-0.00000,  -0.00000,  -0.00000,  12.12500,  -0.00000,   8.37500,  
		11.18750,  14.00000,  26.18750,  -0.00000,  22.43750,  25.25000,  
		28.06250,  40.25000,   1.81250,  36.50000,  39.31250,  42.12500,  
		54.31250,  59.00000, 114.93750, 119.62500, 124.31250, 102.75000,  
		-0.00000,  -0.00000,  -0.00000,  -0.00000,  -0.00000,  21.50000, 
		95.56250,  98.37500, 101.18750, 113.37500,  26.18750, 109.62500, 
		112.43750, 115.25000, 127.43750,  30.87500, 123.68750, 126.50000, 
		129.31250, 141.50000,  35.56250, 137.75000, 140.56250, 143.37500, 
		155.56250, 149.00000, 283.68750, 288.37500, 293.06250, 237.75000};
	
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
	
	CConvolutionalFeatureMap map(w,h,rx,ry,1,1,map_index);
	SGVector<float64_t> params(1+(2*rx+1)*(2*ry+1));
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = CMath::normal_random(0.0,0.01);
	
	input1->compute_activations(x1);
	input2->compute_activations(x2);
	
	SGMatrix<float64_t> A(num_maps*w*h,b);
	SGMatrix<float64_t> buffer(w*h,b);
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

TEST(ConvolutionalFeatureMap, compute_parameter_gradients_with_stride)
{
	const int32_t w = 12;
	const int32_t h = 10;
	const int32_t rx = 1;
	const int32_t ry = 1;
	const int32_t b = 2;
	const int32_t map_index = 1;
	const int32_t num_maps = 3;
	
	int32_t stride_x = 3;
	int32_t stride_y = 2;
	int32_t w_out = w/stride_x;
	int32_t h_out = h/stride_y;
	
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
	
	CConvolutionalFeatureMap map(w,h,rx,ry,stride_x,stride_y,map_index);
	SGVector<float64_t> params(1+(2*rx+1)*(2*ry+1));
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = CMath::normal_random(0.0,0.01);
	
	input1->compute_activations(x1);
	input2->compute_activations(x2);
	
	SGMatrix<float64_t> A(num_maps*w_out*h_out,b);
	SGMatrix<float64_t> buffer(w*h,b);
	A.zero();
	
	map.compute_activations(params, layers, input_indices, A, buffer);
	
	// compute activation gradients with respect to some function
	// assuming the function is 0.5*sum(A[i]^2)
	SGMatrix<float64_t> AG(num_maps*w_out*h_out,b);
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
	
	CConvolutionalFeatureMap map(w,h,rx,ry,1,1,0, CMAF_LOGISTIC);
	SGVector<float64_t> params(1+(2*rx+1)*(2*ry+1));
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
	
	CConvolutionalFeatureMap map(w,h,rx,ry,1,1,0, CMAF_RECTIFIED_LINEAR);
	SGVector<float64_t> params(1+(2*rx+1)*(2*ry+1));
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
	
	CConvolutionalFeatureMap map(w,h,rx,ry,1,1,map_index);
	SGVector<float64_t> params(1+(2*rx+1)*(2*ry+1));
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

TEST(ConvolutionalFeatureMap, pool_activations)
{
	const int32_t w = 6;
	const int32_t h = 4;
	const int32_t pw = 2;
	const int32_t ph = 2;
	const int32_t b = 2;
	const int32_t map_index = 1;
	const int32_t num_maps = 3;
	
	SGMatrix<float64_t> activations(num_maps*w*h,b);
	for (int32_t i=0; i<activations.num_rows*activations.num_cols; i++)
		activations[i] = i;
	
	SGMatrix<float64_t> pooled(num_maps*w*h/(pw*ph),b);
	SGMatrix<float64_t> max_indices(num_maps*w*h/(pw*ph),b);
	
	pooled.zero();
	max_indices.zero();
	
	CConvolutionalFeatureMap map(w,h,1,1,1,1, map_index);
	
	map.pool_activations(activations, pw, ph, pooled, max_indices);
	
	float64_t ref_pooled[] = { 0,0,0,0,0,0,29,31,37,39,45,47,0,0,0,0,0,0,0,0,0,0,
		0,0,101,103,109,111,117,119,0,0,0,0,0,0};
		
	float64_t ref_max_indices[] = { 0,0,0,0,0,0,29,31,37,39,45,47,0,0,0,0,0,0,0,
		0,0,0,0,0,29,31,37,39,45,47,0,0,0,0,0,0};
	
	for (int32_t i=0; i<pooled.num_rows*pooled.num_cols; i++)
		EXPECT_EQ(ref_pooled[i], pooled[i]);
	
	for (int32_t i=0; i<max_indices.num_rows*max_indices.num_cols; i++)
		EXPECT_EQ(ref_max_indices[i], max_indices[i]);
}
