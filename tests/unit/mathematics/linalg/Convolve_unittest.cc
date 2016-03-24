/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Khaled Nasr
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/lib/config.h>

#ifdef HAVE_LINALG_LIB
#include <shogun/mathematics/linalg/linalg.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <gtest/gtest.h>

#include <shogun/mathematics/eigen3.h>

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUMatrix.h>
#include <shogun/lib/GPUVector.h>
#endif

using namespace shogun;

TEST(Convolve, eigen3_backend)
{
	const int32_t w = 6;
	const int32_t h = 5;
	const int32_t sx = 1;
	const int32_t sy = 1;

	SGMatrix<float64_t> X(h,w);
	SGMatrix<float64_t> W(3,3);
	SGMatrix<float64_t> Y(h/sy,w/sx);

	for (int32_t i=0; i<w*h; i++)
		X[i] = i;

	for (float64_t i=0; i<9; i++)
		W[i] = i/4;

	for (int32_t i=0; i<100; i++)
	linalg::convolve<linalg::Backend::EIGEN3>(X, W, Y, false, true, sx, sy);

	// generated with scipy.signal.convolve2d
	float64_t ref[] = {
		2.00000,   6.50000,  10.25000,  14.00000,  14.00000,  13.50000,
		30.00000,  39.00000,  48.00000,  42.00000,  39.75000,  75.00000,
		84.00000,  93.00000,  75.75000,  66.00000, 120.00000, 129.00000,
		138.00000, 109.50000,  92.25000, 165.00000, 174.00000, 183.00000,
		143.25000, 111.00000, 187.25000, 195.50000, 203.75000, 152.00000};

	for (int32_t i=0; i<Y.num_rows*Y.num_cols; i++)
		EXPECT_NEAR(ref[i], Y[i], 1e-15);
}

TEST(Convolve, eigen3_backend_flip)
{
	const int32_t w = 6;
	const int32_t h = 5;
	const int32_t sx = 1;
	const int32_t sy = 1;

	SGMatrix<float64_t> X(h,w);
	SGMatrix<float64_t> W(3,3);
	SGMatrix<float64_t> Y(h/sy,w/sx);

	for (int32_t i=0; i<w*h; i++)
		X[i] = i;

	for (float64_t i=0; i<9; i++)
		W[i] = i/4;

	linalg::convolve<linalg::Backend::EIGEN3>(X, W, Y, true, true, sx, sy);

	// generated with scipy.signal.convolve2d
	float64_t ref[] = {
		22.00000,  35.50000,  43.75000,  52.00000,  34.00000,  52.50000,
		78.00000,  87.00000,  96.00000,  60.00000,  86.25000, 123.00000,
		132.00000, 141.00000,  86.25000, 120.00000, 168.00000, 177.00000,
		186.00000, 112.50000, 153.75000, 213.00000, 222.00000, 231.00000,
		138.75000,  73.00000,  94.75000,  98.50000, 102.25000,  56.00000};

	for (int32_t i=0; i<Y.num_rows*Y.num_cols; i++)
		EXPECT_NEAR(ref[i], Y[i], 1e-15);
}

TEST(Convolve, eigen3_backend_arbitrary_stride)
{
	const int32_t w = 10;
	const int32_t h = 12;
	const int32_t sx = 2;
	const int32_t sy = 3;

	SGMatrix<float64_t> X(h,w);
	SGMatrix<float64_t> W(3,3);
	SGMatrix<float64_t> Y(h/sy,w/sx);

	for (int32_t i=0; i<w*h; i++)
		X[i] = i;

	for (float64_t i=0; i<9; i++)
		W[i] = i/4;

	linalg::convolve<linalg::Backend::EIGEN3>(X, W, Y, false, true, sx, sy);

	// generated with scipy.signal.convolve2d
	float64_t ref[] = {
		3.75000,  19.25000,  30.50000,  41.75000,  92.25000, 187.50000,
		214.50000, 241.50000, 218.25000, 403.50000, 430.50000, 457.50000,
		344.25000, 619.50000, 646.50000, 673.50000, 470.25000, 835.50000,
		862.50000, 889.50000};

	for (int32_t i=0; i<Y.num_rows*Y.num_cols; i++)
		EXPECT_NEAR(ref[i], Y[i], 1e-15);
}

#ifdef HAVE_VIENNACL

TEST(Convolve, viennacl_backend)
{
	const int32_t w = 6;
	const int32_t h = 5;
	const int32_t sx = 1;
	const int32_t sy = 1;

	CGPUMatrix<float64_t> X(h,w);
	CGPUMatrix<float64_t> W(3,3);
	CGPUMatrix<float64_t> Y(h/sy,w/sx);

	for (int32_t i=0; i<w*h; i++)
		X[i] = i;

	for (float64_t i=0; i<9; i++)
		W[i] = i/4;

	linalg::convolve<linalg::Backend::VIENNACL>(X, W, Y, false, true, sx, sy);

	// generated with scipy.signal.convolve2d
	float64_t ref[] = {
		2.00000,   6.50000,  10.25000,  14.00000,  14.00000,  13.50000,
		30.00000,  39.00000,  48.00000,  42.00000,  39.75000,  75.00000,
		84.00000,  93.00000,  75.75000,  66.00000, 120.00000, 129.00000,
		138.00000, 109.50000,  92.25000, 165.00000, 174.00000, 183.00000,
		143.25000, 111.00000, 187.25000, 195.50000, 203.75000, 152.00000};

	for (int32_t i=0; i<Y.num_rows*Y.num_cols; i++)
		EXPECT_NEAR(ref[i], Y[i], 1e-15);
}

TEST(Convolve, viennacl_backend_flip)
{
	const int32_t w = 6;
	const int32_t h = 5;
	const int32_t sx = 1;
	const int32_t sy = 1;

	CGPUMatrix<float64_t> X(h,w);
	CGPUMatrix<float64_t> W(3,3);
	CGPUMatrix<float64_t> Y(h/sy,w/sx);

	for (int32_t i=0; i<w*h; i++)
		X[i] = i;

	for (float64_t i=0; i<9; i++)
		W[i] = i/4;

	linalg::convolve<linalg::Backend::VIENNACL>(X, W, Y, true, true, sx, sy);

	// generated with scipy.signal.convolve2d
	float64_t ref[] = {
		22.00000,  35.50000,  43.75000,  52.00000,  34.00000,  52.50000,
		78.00000,  87.00000,  96.00000,  60.00000,  86.25000, 123.00000,
		132.00000, 141.00000,  86.25000, 120.00000, 168.00000, 177.00000,
		186.00000, 112.50000, 153.75000, 213.00000, 222.00000, 231.00000,
		138.75000,  73.00000,  94.75000,  98.50000, 102.25000,  56.00000};

	for (int32_t i=0; i<Y.num_rows*Y.num_cols; i++)
		EXPECT_NEAR(ref[i], Y[i], 1e-15);
}

TEST(Convolve, viennacl_backend_arbitrary_stride)
{
	const int32_t w = 10;
	const int32_t h = 12;
	const int32_t sx = 2;
	const int32_t sy = 3;

	CGPUMatrix<float64_t> X(h,w);
	CGPUMatrix<float64_t> W(3,3);
	CGPUMatrix<float64_t> Y(h/sy,w/sx);

	for (int32_t i=0; i<w*h; i++)
		X[i] = i;

	for (float64_t i=0; i<9; i++)
		W[i] = i/4;

	linalg::convolve<linalg::Backend::VIENNACL>(X, W, Y, false, true, sx, sy);

	// generated with scipy.signal.convolve2d
	float64_t ref[] = {
		3.75000,  19.25000,  30.50000,  41.75000,  92.25000, 187.50000,
		214.50000, 241.50000, 218.25000, 403.50000, 430.50000, 457.50000,
		344.25000, 619.50000, 646.50000, 673.50000, 470.25000, 835.50000,
		862.50000, 889.50000};

	for (int32_t i=0; i<Y.num_rows*Y.num_cols; i++)
		EXPECT_NEAR(ref[i], Y[i], 1e-15);
}

#endif // HAVE_VIENNACL

#endif // HAVE_LINALG_LIB
