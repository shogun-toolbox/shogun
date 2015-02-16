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

#if defined(HAVE_CXX0X) || defined(HAVE_CXX11)
#include <shogun/mathematics/linalg/linalg.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <gtest/gtest.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUMatrix.h>
#include <shogun/lib/GPUVector.h>
#endif

using namespace shogun;

TEST(AddMatrix, native_backend)
{
	const float64_t alpha = 0.3;
	const float64_t beta = -1.5;
	
	SGMatrix<float64_t> A(3,3);
	SGMatrix<float64_t> B(3,3);
	SGMatrix<float64_t> C(3,3);
	SGMatrix<float64_t> D(3,3);
	
	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}
	
	linalg::add<linalg::Backend::NATIVE>(A, B, C, alpha, beta);
	linalg::add<linalg::Backend::NATIVE>(A, B, D);
	
	for (int32_t i=0; i<9; i++)
	{
		EXPECT_NEAR(alpha*A[i]+beta*B[i], C[i], 1e-15);
		EXPECT_NEAR(A[i]+B[i], D[i], 1e-15);
	}
}

TEST(AddVector, native_backend)
{
	const float64_t alpha = 0.3;
	const float64_t beta = -1.5;
	
	SGVector<float64_t> A(9);
	SGVector<float64_t> B(9);
	SGVector<float64_t> C(9);
	SGVector<float64_t> D(9);
	
	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}
	
	linalg::add<linalg::Backend::NATIVE>(A, B, C, alpha, beta);
	linalg::add<linalg::Backend::NATIVE>(A, B, D);
	
	for (int32_t i=0; i<9; i++)
	{
		EXPECT_NEAR(alpha*A[i]+beta*B[i], C[i], 1e-15);
		EXPECT_NEAR(A[i]+B[i], D[i], 1e-15);
	}
}

#ifdef HAVE_LINALG_LIB
#ifdef HAVE_EIGEN3
TEST(AddMatrix, eigen3_backend)
{
	const float64_t alpha = 0.3;
	const float64_t beta = -1.5;
	
	SGMatrix<float64_t> A(3,3);
	SGMatrix<float64_t> B(3,3);
	SGMatrix<float64_t> C(3,3);
	
	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}
	
	linalg::add<linalg::Backend::EIGEN3>(A, B, C, alpha, beta);
	
	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(alpha*A[i]+beta*B[i], C[i], 1e-15);
}

TEST(AddVector, eigen3_backend)
{
	const float64_t alpha = 0.3;
	const float64_t beta = -1.5;
	
	SGVector<float64_t> A(9);
	SGVector<float64_t> B(9);
	SGVector<float64_t> C(9);
	
	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}
	
	linalg::add<linalg::Backend::EIGEN3>(A, B, C, alpha, beta);
	
	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(alpha*A[i]+beta*B[i], C[i], 1e-15);
}
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
TEST(AddMatrix, viennacl_backend)
{
	const float64_t alpha = 0.3;
	const float64_t beta = -1.5;
	
	CGPUMatrix<float64_t> A(3,3);
	CGPUMatrix<float64_t> B(3,3);
	CGPUMatrix<float64_t> C(3,3);
	
	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}
	
	linalg::add<linalg::Backend::VIENNACL>(A, B, C, alpha, beta);
	
	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(alpha*A[i]+beta*B[i], C[i], 1e-15);
}

TEST(AddVector, viennacl_backend)
{
	const float64_t alpha = 0.3;
	const float64_t beta = -1.5;
	
	CGPUVector<float64_t> A(9);
	CGPUVector<float64_t> B(9);
	CGPUVector<float64_t> C(9);
	
	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}
	
	linalg::add<linalg::Backend::VIENNACL>(A, B, C, alpha, beta);
	
	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(alpha*A[i]+beta*B[i], C[i], 1e-15);
}
#endif // HAVE_VIENNACL
#endif // HAVE_LINALG_LIB

#endif // defined(HAVE_CXX0X) || defined(HAVE_CXX11)
