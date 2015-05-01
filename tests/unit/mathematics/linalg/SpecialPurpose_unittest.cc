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
#include <shogun/mathematics/Math.h>
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
TEST(SpecialPurpose, logistic_eigen3_backend)
{
	SGMatrix<float64_t> A(3,3);
	SGMatrix<float64_t> B(3,3);

	for (int32_t i=0; i<9; i++)
		A[i] = i;

	linalg::special_purpose::logistic<linalg::Backend::EIGEN3>(A, B);

	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(1.0/(1+CMath::exp(-1*A[i])), B[i], 1e-15);
}
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
TEST(SpecialPurpose, logistic_viennacl_backend)
{
	CGPUMatrix<float64_t> A(3,3);
	CGPUMatrix<float64_t> B(3,3);

	for (int32_t i=0; i<9; i++)
		A[i] = i;

	linalg::special_purpose::logistic<linalg::Backend::VIENNACL>(A, B);

	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(1.0/(1+CMath::exp(-1*A[i])), B[i], 1e-15);
}
#endif // HAVE_VIENNACL

#ifdef HAVE_EIGEN3
TEST(SpecialPurpose, multiply_by_logistic_derivative_eigen3_backend)
{
	SGMatrix<float64_t> A(3,3);
	SGMatrix<float64_t> B(3,3);

	for (float64_t i=0; i<9; i++)
	{
		A[i] = i/9;
		B[i] = i;
	}

	linalg::special_purpose::multiply_by_logistic_derivative<linalg::Backend::EIGEN3>(A, B);

	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(i*A[i]*(1.0-A[i]), B[i], 1e-15);
}
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
TEST(SpecialPurpose, multiply_by_logistic_derivative_viennacl_backend)
{
	CGPUMatrix<float64_t> A(3,3);
	CGPUMatrix<float64_t> B(3,3);

	for (float64_t i=0; i<9; i++)
	{
		A[i] = i/9;
		B[i] = i;
	}

	linalg::special_purpose::multiply_by_logistic_derivative<linalg::Backend::VIENNACL>(A, B);

	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(i*A[i]*(1.0-A[i]), B[i], 1e-15);
}
#endif // HAVE_VIENNACL

#ifdef HAVE_EIGEN3
TEST(SpecialPurpose, rectified_linear_eigen3_backend)
{
	SGMatrix<float64_t> A(3,3);
	SGMatrix<float64_t> B(3,3);

	for (int32_t i=0; i<9; i++)
		A[i] = i-5;

	linalg::special_purpose::rectified_linear<linalg::Backend::EIGEN3>(A, B);

	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(CMath::max(0.0,A[i]), B[i], 1e-15);
}
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
TEST(SpecialPurpose, rectified_linear_viennacl_backend)
{
	CGPUMatrix<float64_t> A(3,3);
	CGPUMatrix<float64_t> B(3,3);

	for (int32_t i=0; i<9; i++)
		A[i] = i-5;

	linalg::special_purpose::rectified_linear<linalg::Backend::VIENNACL>(A, B);

	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(CMath::max(0.0, (float64_t)A[i]), B[i], 1e-15);
}
#endif // HAVE_VIENNACL

#ifdef HAVE_EIGEN3
TEST(SpecialPurpose, multiply_by_rectified_linear_derivative_eigen3_backend)
{
	SGMatrix<float64_t> A(3,3);
	SGMatrix<float64_t> B(3,3);

	for (float64_t i=0; i<9; i++)
	{
		A[i] = i - 0.5;
		B[i] = i;
	}

	linalg::special_purpose::multiply_by_rectified_linear_derivative<linalg::Backend::EIGEN3>(A, B);

	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(i*(A[i]!=0), B[i], 1e-15);
}
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
TEST(SpecialPurpose, multiply_by_rectified_linear_derivative_viennacl_backend)
{
	CGPUMatrix<float64_t> A(3,3);
	CGPUMatrix<float64_t> B(3,3);

	for (float64_t i=0; i<9; i++)
	{
		A[i] = i - 0.5;
		B[i] = i;
	}

	linalg::special_purpose::multiply_by_rectified_linear_derivative<linalg::Backend::VIENNACL>(A, B);

	for (int32_t i=0; i<9; i++)
		EXPECT_NEAR(i*(A[i]!=0), B[i], 1e-15);
}
#endif // HAVE_VIENNACL

#ifdef HAVE_EIGEN3
TEST(SpecialPurpose, softmax_eigen3_backend)
{
	SGMatrix<float64_t> A(4,3);

	SGMatrix<float64_t> ref(4,3);

	for (float64_t i=0; i<12; i++)
		A[i] = i/12;

	for (int32_t i=0; i<A.num_rows*A.num_cols; i++)
		ref[i] = CMath::exp(A[i]);

	for (int32_t j=0; j<ref.num_cols; j++)
	{
		float64_t sum = 0;
		for (int32_t i=0; i<ref.num_rows; i++)
			sum += ref(i,j);

		for (int32_t i=0; i<ref.num_rows; i++)
			ref(i,j) /= sum;
	}

	linalg::special_purpose::softmax<linalg::Backend::EIGEN3>(A);

	for (int32_t i=0; i<12; i++)
		EXPECT_NEAR(ref[i], A[i], 1e-15);
}
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
TEST(SpecialPurpose, softmax_viennacl_backend)
{
	CGPUMatrix<float64_t> A(4,3);

	CGPUMatrix<float64_t> ref(4,3);

	for (float64_t i=0; i<12; i++)
		A[i] = i/12;

	for (int32_t i=0; i<A.num_rows*A.num_cols; i++)
		ref[i] = CMath::exp(A[i]);

	for (int32_t j=0; j<ref.num_cols; j++)
	{
		float64_t sum = 0;
		for (int32_t i=0; i<ref.num_rows; i++)
			sum += ref(i,j);

		for (int32_t i=0; i<ref.num_rows; i++)
			ref(i,j) /= sum;
	}

	linalg::special_purpose::softmax<linalg::Backend::VIENNACL>(A);

	for (int32_t i=0; i<12; i++)
		EXPECT_NEAR(ref[i], A[i], 1e-15);
}
#endif // HAVE_VIENNACL

#ifdef HAVE_EIGEN3
TEST(SpecialPurpose, cross_entropy_eigen3_backend)
{
	SGMatrix<float64_t> A(4,3);
	SGMatrix<float64_t> B(4,3);

	int32_t size = A.num_rows*A.num_cols;
	for (float64_t i=0; i<size; i++)
	{
		A[i] = i/size;
		B[i] = (i/size) * 0.5;
	}

	float64_t ce = 0;
	for (int32_t i=0; i< size; i++)
		ce += A[i]*CMath::log(B[i]+1e-30);
	ce *= -1;

	EXPECT_NEAR(ce, linalg::special_purpose::cross_entropy<linalg::Backend::EIGEN3>(A, B), 1e-15);
}
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
TEST(SpecialPurpose, cross_entropy_viennacl_backend)
{
	CGPUMatrix<float64_t> A(4,3);
	CGPUMatrix<float64_t> B(4,3);

	int32_t size = A.num_rows*A.num_cols;
	for (float64_t i=0; i<size; i++)
	{
		A[i] = i/size;
		B[i] = (i/size) * 0.5;
	}

	float64_t ce = 0;
	for (int32_t i=0; i< size; i++)
		ce += A[i]*CMath::log(B[i]+1e-30);
	ce *= -1;

	EXPECT_NEAR(ce, linalg::special_purpose::cross_entropy<linalg::Backend::VIENNACL>(A, B), 1e-15);
}
#endif // HAVE_VIENNACL

#ifdef HAVE_EIGEN3
TEST(SpecialPurpose, squared_error_eigen3_backend)
{
	SGMatrix<float64_t> A(4,3);
	SGMatrix<float64_t> B(4,3);

	int32_t size = A.num_rows*A.num_cols;
	for (float64_t i=0; i<size; i++)
	{
		A[i] = i/size;
		B[i] = (i/size) * 0.5;
	}

	float64_t se = 0;
	for (int32_t i=0; i< size; i++)
		se += CMath::pow(A[i]-B[i],2);
	se *= 0.5;

	EXPECT_NEAR(se, linalg::special_purpose::squared_error<linalg::Backend::EIGEN3>(A, B), 1e-15);
}
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
TEST(SpecialPurpose, squared_error_viennacl_backend)
{
	CGPUMatrix<float64_t> A(4,3);
	CGPUMatrix<float64_t> B(4,3);

	int32_t size = A.num_rows*A.num_cols;
	for (float64_t i=0; i<size; i++)
	{
		A[i] = i/size;
		B[i] = (i/size) * 0.5;
	}

	float64_t se = 0;
	for (int32_t i=0; i< size; i++)
		se += CMath::pow(A[i]-B[i],2);
	se *= 0.5;

	EXPECT_NEAR(se, linalg::special_purpose::squared_error<linalg::Backend::VIENNACL>(A, B), 1e-15);
}
#endif // HAVE_VIENNACL

#endif // HAVE_LINALG_LIB
