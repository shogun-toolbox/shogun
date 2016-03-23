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

#include <shogun/mathematics/eigen3.h>

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUMatrix.h>
#endif

using namespace shogun;

TEST(ElementwiseProduct, SGMatrix_eigen3_backend)
{
	SGMatrix<float64_t> A(3,3);
	SGMatrix<float64_t> B(3,3);
	SGMatrix<float64_t> C(3,3);

	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}

	linalg::elementwise_product<linalg::Backend::EIGEN3>(A, B, C);

	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(A[i]*B[i], C[i], 1e-15);
}

#ifdef HAVE_VIENNACL
TEST(ElementwiseProduct, CGPUMatrix_eigen3_backend)
{
	CGPUMatrix<float64_t> A(3,3);
	CGPUMatrix<float64_t> B(3,3);

	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}

	CGPUMatrix<float64_t> C = linalg::elementwise_product<linalg::Backend::EIGEN3>(A, B);

	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(A[i]*B[i], C[i], 1e-15);
}

TEST(ElementwiseProduct, CGPUMatrix_viennacl_backend)
{
	CGPUMatrix<float64_t> A(3,3);
	CGPUMatrix<float64_t> B(3,3);
	CGPUMatrix<float64_t> C(3,3);

	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}

	linalg::elementwise_product<linalg::Backend::VIENNACL>(A, B, C);

	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(A[i]*B[i], C[i], 1e-15);
}

TEST(ElementwiseProduct, SGMatrix_viennacl_backend)
{
	SGMatrix<float64_t> A(3,3);
	SGMatrix<float64_t> B(3,3);

	for (int32_t i=0; i<9; i++)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}

	SGMatrix<float64_t> C = linalg::elementwise_product<linalg::Backend::VIENNACL>(A, B);

	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(A[i]*B[i], C[i], 1e-15);
}
#endif // HAVE_VIENNACL

#endif // HAVE_LINALG_LIB
