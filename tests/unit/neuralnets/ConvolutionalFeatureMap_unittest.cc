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
	SGVector<float64_t> params(1+(2*rx+1)*(2*ry+1)*3);
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = float64_t(i)/4;

	input1->compute_activations(x1);
	input2->compute_activations(x2);

	SGMatrix<float64_t> A(num_maps*w*h,b);
	map.compute_activations(params, layers, input_indices, A);

	// reference numbers generated using scipy.signal.convolve2d
	float64_t ref[] = {
		95.12500, 153.34375, 164.96875, 176.59375, 127.75000, 181.59375,
		294.00000, 315.65625, 337.31250, 243.65625, 249.09375, 402.28125,
		423.93750, 445.59375, 320.53125, 316.59375, 510.56250, 532.21875,
		553.87500, 397.40625, 384.09375, 618.84375, 640.50000, 662.15625,
		474.28125, 335.12500, 534.90625, 552.15625, 569.40625, 404.00000,
		432.62500, 693.34375, 704.96875, 716.59375, 510.25000, 789.09375,
		1255.87500,1277.53125,1299.18750, 918.65625, 856.59375,1364.15625,
		1385.81250,1407.46875, 995.53125, 924.09375,1472.43750,1494.09375,
		1515.75000,1072.40625, 991.59375,1580.71875,1602.37500,1624.03125,
		1149.28125, 807.62500,1277.40625,1294.65625,1311.90625, 921.50000};

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
	SGVector<float64_t> params(1+(2*rx+1)*(2*ry+1)*3);
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = float64_t(i)/4;

	input1->compute_activations(x1);
	input2->compute_activations(x2);

	SGMatrix<float64_t> A(num_maps*w_out*h_out,b);
	map.compute_activations(params, layers, input_indices, A);

	// reference numbers generated using scipy.signal.convolve2d
	float64_t ref[] = {
		344.50000, 549.81250, 573.06250, 596.31250, 619.56250, 880.03125,
		1411.12500,1454.43750,1497.75000,1541.06250,1285.03125,2060.81250,
		2104.12500,2147.43750,2190.75000,1690.03125,2710.50000,2753.81250,
		2797.12500,2840.43750,1694.50000,2709.81250,2733.06250,2756.31250,
		2779.56250,3310.03125,5258.62500,5301.93750,5345.25000,5388.56250,
		3715.03125,5908.31250,5951.62500,5994.93750,6038.25000,4120.03125,
		6558.00000,6601.31250,6644.62500,6687.93750};

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
	SGVector<float64_t> params(1+(2*rx+1)*(2*ry+1)*3);
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = float64_t(i)*1e-4;

	input1->compute_activations(x1);
	input2->compute_activations(x2);

	SGMatrix<float64_t> A(num_maps*w*h,b);
	map.compute_activations(params, layers, input_indices, A);

	float64_t ref[] = {
		0.50951,   0.51533,   0.51649,   0.51765,   0.51277,   0.51815,
		0.52937,   0.53152,   0.53368,   0.52435,   0.52489,   0.54014,
		0.54229,   0.54444,   0.53201,   0.53162,   0.55088,   0.55302,
		0.55516,   0.53966,   0.53833,   0.56157,   0.56370,   0.56583,
		0.54729,   0.53346,   0.55329,   0.55499,   0.55670,   0.54031,
		0.54315,   0.56889,   0.57003,   0.57117,   0.55085,   0.57826,
		0.62301,   0.62504,   0.62707,   0.59085,   0.58483,   0.63313,
		0.63514,   0.63714,   0.59826,   0.59137,   0.64313,   0.64512,
		0.64710,   0.60563,   0.59788,   0.65301,   0.65497,   0.65692,
		0.61295,   0.58007,   0.62503,   0.62665,   0.62826,   0.59112};

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
	SGVector<float64_t> params(1+(2*rx+1)*(2*ry+1)*3);
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = float64_t(i)/4 - 2.5;

	input1->compute_activations(x1);
	input2->compute_activations(x2);

	SGMatrix<float64_t> A(num_maps*w*h,b);
	map.compute_activations(params, layers, input_indices, A);

	float64_t ref[] = {
		17.62500,  28.96875,  21.84375,  14.71875,  12.75000,  19.71875,
		38.37500,  31.90625,  25.43750,  25.53125,  -0.00000,   6.03125,
		-0.00000,  -0.00000,   8.65625,  -0.00000,  -0.00000,  -0.00000,
		-0.00000,  -0.00000,  -0.00000,  -0.00000,  -0.00000,  -0.00000,
		-0.00000,   7.62500,  35.53125,  34.03125,  32.53125,  39.00000,
		-0.00000,  -0.00000,  -0.00000,  -0.00000,  -0.00000,  -0.00000,
		-0.00000,  -0.00000,  -0.00000,  25.53125,  -0.00000,  -0.00000,
		-0.00000,  -0.00000,   8.65625,  -0.00000,  -0.00000,  -0.00000,
		-0.00000,  -0.00000,  -0.00000,  -0.00000,  -0.00000,  -0.00000,
		-0.00000,  30.12500, 103.03125, 101.53125, 100.03125, 106.50000};

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
	SGVector<float64_t> params(1+(2*rx+1)*(2*ry+1)*3);
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = CMath::normal_random(0.0,0.01);

	input1->compute_activations(x1);
	input2->compute_activations(x2);

	SGMatrix<float64_t> A(num_maps*w*h,b);
	A.zero();

	map.compute_activations(params, layers, input_indices, A);

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
		map.compute_activations(params, layers, input_indices, A);
		float64_t error_plus = 0;
		for (int32_t k=0; k<A.num_rows*A.num_cols; k++)
			error_plus += 0.5*A[k]*A[k];

		params[i] -= 2*epsilon;
		map.compute_activations(params, layers, input_indices, A);
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
	SGVector<float64_t> params(1+(2*rx+1)*(2*ry+1)*3);
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = CMath::normal_random(0.0,0.01);

	input1->compute_activations(x1);
	input2->compute_activations(x2);

	SGMatrix<float64_t> A(num_maps*w_out*h_out,b);
	A.zero();

	map.compute_activations(params, layers, input_indices, A);

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
		map.compute_activations(params, layers, input_indices, A);
		float64_t error_plus = 0;
		for (int32_t k=0; k<A.num_rows*A.num_cols; k++)
			error_plus += 0.5*A[k]*A[k];

		params[i] -= 2*epsilon;
		map.compute_activations(params, layers, input_indices, A);
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
	A.zero();

	map.compute_activations(params, layers, input_indices, A);

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
		map.compute_activations(params, layers, input_indices, A);
		float64_t error_plus = 0;
		for (int32_t k=0; k<A.num_rows*A.num_cols; k++)
			error_plus += 0.5*A[k]*A[k];

		params[i] -= 2*epsilon;
		map.compute_activations(params, layers, input_indices, A);
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
	A.zero();

	map.compute_activations(params, layers, input_indices, A);

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
		map.compute_activations(params, layers, input_indices, A);
		float64_t error_plus = 0;
		for (int32_t k=0; k<A.num_rows*A.num_cols; k++)
			error_plus += 0.5*A[k]*A[k];

		params[i] -= 2*epsilon;
		map.compute_activations(params, layers, input_indices, A);
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
	SGVector<float64_t> params(1+(2*rx+1)*(2*ry+1)*3);
	for (int32_t i=0; i<params.vlen; i++)
		params[i] = CMath::normal_random(0.0,0.01);

	SGMatrix<float64_t> A(num_maps*w*h,b);
	A.zero();

	map.compute_activations(params, layers, input_indices, A);

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
		map.compute_activations(params, layers, input_indices, A);
		float64_t error_plus = 0;
		for (int32_t k=0; k<A.num_rows*A.num_cols; k++)
			error_plus += 0.5*A[k]*A[k];

		input1->get_activations()[i] -= 2*epsilon;
		map.compute_activations(params, layers, input_indices, A);
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
		map.compute_activations(params, layers, input_indices, A);
		float64_t error_plus = 0;
		for (int32_t k=0; k<A.num_rows*A.num_cols; k++)
			error_plus += 0.5*A[k]*A[k];

		input2->get_activations()[i] -= 2*epsilon;
		map.compute_activations(params, layers, input_indices, A);
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
