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
#include <gtest/gtest.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUMatrix.h>
#endif

using namespace shogun;

#ifdef HAVE_EIGEN3
TEST(MatrixProduct, eigen3_backend)
{
	SGMatrix<float64_t> A(3,3);
	SGMatrix<float64_t> B(3,3);
	SGMatrix<float64_t> C(3,3);
	
	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}
	
	linalg::matrix_product<linalg::Backend::EIGEN3>(A, B, C);
	
	float64_t ref[] = {7.5, 9, 10.5, 21, 27, 33, 34.5, 45, 55.5};
	
	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(ref[i], C[i], 1e-15);
}

TEST(MatrixProduct, eigen3_backend_transpose_A)
{
	SGMatrix<float64_t> A(3,3);
	SGMatrix<float64_t> B(3,3);
	SGMatrix<float64_t> C(3,3);
	
	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}
	
	linalg::matrix_product<linalg::Backend::EIGEN3>(A, B, C, true);
	
	float64_t ref[] = {2.5, 7, 11.5, 7, 25, 43, 11.5, 43, 74.5};
	
	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(ref[i], C[i], 1e-15);
}

TEST(MatrixProduct, eigen3_backend_transpose_B)
{
	SGMatrix<float64_t> A(3,3);
	SGMatrix<float64_t> B(3,3);
	SGMatrix<float64_t> C(3,3);
	
	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}
	
	linalg::matrix_product<linalg::Backend::EIGEN3>(A, B, C, false, true);
	
	float64_t ref[] = {22.5, 27, 31.5, 27, 33, 39, 31.5, 39, 46.5};
	
	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(ref[i], C[i], 1e-15);
}

TEST(MatrixProduct, eigen3_backend_transpose_A_transpose_B)
{
	SGMatrix<float64_t> A(3,3);
	SGMatrix<float64_t> B(3,3);
	SGMatrix<float64_t> C(3,3);
	
	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}
	
	linalg::matrix_product<linalg::Backend::EIGEN3>(A, B, C, true, true);
	
	float64_t ref[] = {7.5, 21, 34.5, 9, 27, 45, 10.5, 33, 55.5};
	
	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(ref[i], C[i], 1e-15);
}

#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
TEST(MatrixProduct, viennacl_backend)
{
	CGPUMatrix<float64_t> A(3,3);
	CGPUMatrix<float64_t> B(3,3);
	CGPUMatrix<float64_t> C(3,3);
	
	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}
	
	linalg::matrix_product<linalg::Backend::VIENNACL>(A, B, C);
	
	float64_t ref[] = {7.5, 9, 10.5, 21, 27, 33, 34.5, 45, 55.5};
	
	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(ref[i], C[i], 1e-15);
}

TEST(MatrixProduct, viennacl_backend_transpose_A)
{
	CGPUMatrix<float64_t> A(3,3);
	CGPUMatrix<float64_t> B(3,3);
	CGPUMatrix<float64_t> C(3,3);
	
	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}
	
	linalg::matrix_product<linalg::Backend::VIENNACL>(A, B, C, true);
	
	float64_t ref[] = {2.5, 7, 11.5, 7, 25, 43, 11.5, 43, 74.5};
	
	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(ref[i], C[i], 1e-15);
}

TEST(MatrixProduct, viennacl_backend_transpose_B)
{
	CGPUMatrix<float64_t> A(3,3);
	CGPUMatrix<float64_t> B(3,3);
	CGPUMatrix<float64_t> C(3,3);
	
	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}
	
	linalg::matrix_product<linalg::Backend::VIENNACL>(A, B, C, false, true);
	
	float64_t ref[] = {22.5, 27, 31.5, 27, 33, 39, 31.5, 39, 46.5};
	
	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(ref[i], C[i], 1e-15);
}

TEST(MatrixProduct, viennacl_backend_transpose_A_transpose_B)
{
	CGPUMatrix<float64_t> A(3,3);
	CGPUMatrix<float64_t> B(3,3);
	CGPUMatrix<float64_t> C(3,3);
	
	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}
	
	linalg::matrix_product<linalg::Backend::VIENNACL>(A, B, C, true, true);
	
	float64_t ref[] = {7.5, 21, 34.5, 9, 27, 45, 10.5, 33, 55.5};
	
	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(ref[i], C[i], 1e-15);
}

#endif // HAVE_VIENNACL

#endif // HAVE_LINALG_LIB
