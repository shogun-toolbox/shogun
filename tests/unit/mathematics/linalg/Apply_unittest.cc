/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Soumyajit De
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
#endif // HAVE_VIENNACL

using namespace shogun;

TEST(Apply, SGMatrix_Eigen3_backend)
{
	const index_t rows=4;
	const index_t cols=3;

	SGMatrix<float64_t> A(rows, cols);
	SGVector<float64_t> b(cols);
	SGVector<float64_t> x(rows);

	for (index_t i=0; i<cols; i++)
	{
		for (index_t j=0; j<rows; ++j)
			A(j, i)=i*rows+j;
		b[i]=0.5*i;
	}

	linalg::apply<linalg::Backend::EIGEN3>(A, b, x);

	float64_t ref[] = {10,11.5,13,14.5};

	for (index_t i=0; i<x.vlen; ++i)
		EXPECT_NEAR(x[i], ref[i], 1e-15);
}

TEST(Apply, SGMatrix_Eigen3_backend_transpose)
{
	const index_t rows=3;
	const index_t cols=4;

	SGMatrix<float64_t> A(rows, cols);
	SGVector<float64_t> b(rows);
	SGVector<float64_t> x(cols);

	for (index_t i=0; i<rows; i++)
	{
		for (index_t j=0; j<cols; ++j)
			A(i, j)=i*cols+j;
		b[i]=0.5*i;
	}

	linalg::apply<linalg::Backend::EIGEN3>(A, b, x, true);

	float64_t ref[] = {10,11.5,13,14.5};

	for (index_t i=0; i<x.vlen; ++i)
		EXPECT_NEAR(x[i], ref[i], 1e-15);
}

#ifdef HAVE_VIENNACL
TEST(Apply, CGPUMatrix_Eigen3_backend)
{
	const index_t rows=4;
	const index_t cols=3;

	CGPUMatrix<float64_t> A(rows, cols);
	CGPUVector<float64_t> b(cols);

	for (index_t i=0; i<cols; i++)
	{
		for (index_t j=0; j<rows; ++j)
			A(j, i)=i*rows+j;
		b[i]=0.5*i;
	}

	CGPUVector<float64_t> x=linalg::apply<linalg::Backend::EIGEN3>(A, b);

	float64_t ref[] = {10,11.5,13,14.5};

	for (index_t i=0; i<x.vlen; ++i)
		EXPECT_NEAR(x[i], ref[i], 1e-15);
}

TEST(Apply, CGPUMatrix_ViennaCL_backend)
{
	const index_t rows=4;
	const index_t cols=3;

	CGPUMatrix<float64_t> A(rows, cols);
	CGPUVector<float64_t> b(cols);
	CGPUVector<float64_t> x(rows);

	for (index_t i=0; i<cols; i++)
	{
		for (index_t j=0; j<rows; ++j)
			A(j, i)=i*rows+j;
		b[i]=0.5*i;
	}

	linalg::apply<linalg::Backend::VIENNACL>(A, b, x);

	float64_t ref[] = {10,11.5,13,14.5};

	for (index_t i=0; i<x.vlen; ++i)
		EXPECT_NEAR(x[i], ref[i], 1e-15);
}

TEST(Apply, CGPUMatrix_ViennaCL_backend_transpose)
{
	const index_t rows=3;
	const index_t cols=4;

	CGPUMatrix<float64_t> A(rows, cols);
	CGPUVector<float64_t> b(rows);
	CGPUVector<float64_t> x(cols);

	for (index_t i=0; i<rows; i++)
	{
		for (index_t j=0; j<cols; ++j)
			A(i, j)=i*cols+j;
		b[i]=0.5*i;
	}

	linalg::apply<linalg::Backend::VIENNACL>(A, b, x, true);

	float64_t ref[] = {10,11.5,13,14.5};

	for (index_t i=0; i<x.vlen; ++i)
		EXPECT_NEAR(x[i], ref[i], 1e-15);
}

TEST(Apply, SGMatrix_ViennaCL_backend)
{
	const index_t rows=4;
	const index_t cols=3;

	SGMatrix<float64_t> A(rows, cols);
	SGVector<float64_t> b(cols);

	for (index_t i=0; i<cols; i++)
	{
		for (index_t j=0; j<rows; ++j)
			A(j, i)=i*rows+j;
		b[i]=0.5*i;
	}

	SGVector<float64_t> x=linalg::apply<linalg::Backend::VIENNACL>(A, b);

	float64_t ref[] = {10,11.5,13,14.5};

	for (index_t i=0; i<x.vlen; ++i)
		EXPECT_NEAR(x[i], ref[i], 1e-15);
}
#endif // HAVE_VIENNACL
#endif // HAVE_LINALG_LIB
