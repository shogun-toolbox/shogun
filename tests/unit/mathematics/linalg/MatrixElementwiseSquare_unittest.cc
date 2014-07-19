/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
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
#include <shogun/mathematics/Math.h>
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
TEST(MatrixElementwiseSquare, SGMatrix_eigen3_backend)
{
	const index_t m=2;
	const index_t n=3;
	SGMatrix<float64_t> mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGMatrix<float64_t> sq=linalg::elementwise_square<linalg::Backend::EIGEN3>(mat);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			EXPECT_NEAR(sq(i,j), CMath::sq(mat(i,j)), 1E-15);
	}
}

TEST(MatrixElementwiseSquare, Eigen3_Matrix_eigen3_backend)
{
	const index_t m=2;
	const index_t n=3;
	Eigen::MatrixXd mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGMatrix<float64_t> sq=linalg::elementwise_square<linalg::Backend::EIGEN3>(mat);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			EXPECT_NEAR(sq(i,j), CMath::sq(mat(i,j)), 1E-15);
	}
}

TEST(MatrixElementwiseSquare, SGMatrix_block_eigen3_backend)
{
	const index_t m=2;
	const index_t n=3;
	SGMatrix<float64_t> mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGMatrix<float64_t> sq=linalg::elementwise_square<linalg::Backend::EIGEN3>(
		linalg::block(mat,0,0,2,2));

	for (index_t i=0; i<2; ++i)
	{
		for (index_t j=0; j<2; ++j)
			EXPECT_NEAR(sq(i,j), CMath::sq(mat(i,j)), 1E-15);
	}
}

TEST(MatrixElementwiseSquare, Eigen3_block_eigen3_backend)
{
	const index_t m=2;
	const index_t n=3;
	Eigen::MatrixXd mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGMatrix<float64_t> sq=linalg::elementwise_square<linalg::Backend::EIGEN3>(
		linalg::block((SGMatrix<float64_t>)mat,0,0,2,2));

	for (index_t i=0; i<2; ++i)
	{
		for (index_t j=0; j<2; ++j)
			EXPECT_NEAR(sq(i,j), CMath::sq(mat(i,j)), 1E-15);
	}
}
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL

#endif // HAVE_VIENNACL

TEST(MatrixElementwiseSquare, viennacl_backend)
{
	const index_t m=2;
	const index_t n=3;
	CGPUMatrix<float64_t> mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	CGPUMatrix<float64_t> sq=linalg::elementwise_square<linalg::Backend::VIENNACL>(mat);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			EXPECT_NEAR(sq(i,j), mat(i,j)*mat(i,j), 1E-15);
	}
}

#endif // HAVE_LINALG_LIB
