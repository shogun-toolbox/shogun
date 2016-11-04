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

#include <shogun/lib/config.h>

#ifdef HAVE_VIENNACL
#ifdef HAVE_CXX11

#include <shogun/lib/GPUMatrix.h>
#include <viennacl/matrix.hpp>
#include <viennacl/linalg/prod.hpp>
#include <gtest/gtest.h>
#include <shogun/mathematics/eigen3.h>

#include <shogun/lib/SGMatrix.h>

using namespace shogun;

TEST(GPUMatrix, element_read_write_parentheses_operator)
{
	const int nrows = 3;
	const int ncols = 4;

	CGPUMatrix<float64_t> mat(nrows,ncols);

	for (int32_t i=0; i<nrows; i++)
	{
		for (int32_t j=0; j<ncols; j++)
			mat(i,j) = i + j*nrows;
	}

	for (int32_t i=0; i<nrows; i++)
	{
		for (int32_t j=0; j<ncols; j++)
			EXPECT_EQ(i+j*nrows, mat(i,j));
	}
}

TEST(GPUMatrix, element_read_write_brackets_operator)
{
	const int nrows = 3;
	const int ncols = 4;

	CGPUMatrix<float64_t> mat(nrows,ncols);

	for (int32_t i=0; i<nrows*ncols; i++)
		mat[i] = i;

	for (int32_t i=0; i<nrows*ncols; i++)
		EXPECT_EQ(i, mat[i]);
}

TEST(GPUMatrix, zero)
{
	const int nrows = 3;
	const int ncols = 4;

	CGPUMatrix<float64_t> mat(nrows,ncols);
	mat.zero();

	for (int32_t i=0; i<nrows*ncols; i++)
		EXPECT_EQ(0, mat[i]);
}

TEST(GPUMatrix, set_const)
{
	const int nrows = 3;
	const int ncols = 4;

	CGPUMatrix<float64_t> mat(nrows,ncols);
	mat.set_const(3);

	for (int32_t i=0; i<nrows*ncols; i++)
		EXPECT_EQ(3, mat[i]);
}

/** Tests element access operations on a matrix that was created with an offset */
TEST(GPUMatrix, element_access_with_offset)
{
	CGPUMatrix<float64_t> data(5,5);
	for (int32_t i=0; i<5*5; i++)
		data[i] = i;

	CGPUMatrix<float64_t> mat(data.matrix, 3, 4, 7);

	for (int32_t i=0; i<3*4; i++)
		EXPECT_EQ(data[i+7], mat[i]);
}

/** Tests matrix multiplication performed using ViennaCL on two CGPUMatrix objects
 * that were created with offsets
 */
TEST(GPUMatrix, matrix_multiplication_with_offset)
{
	CGPUMatrix<float64_t> data(6,6);
	for (int32_t i=0; i<36; i++)
		data[i] = i;

	CGPUMatrix<float64_t> A(data.matrix, 3, 4, 0);
	CGPUMatrix<float64_t> B(data.matrix, 4, 6, 12);

	CGPUMatrix<float64_t> C(3,6);

	C.vcl_matrix() = viennacl::linalg::prod(A.vcl_matrix(), B.vcl_matrix());

	SGMatrix<float64_t> C_sg = SGMatrix<float64_t>::matrix_multiply(A, B);

	for (int32_t i=0; i<3*6; i++)
		EXPECT_NEAR(C_sg[i], C[i], 1e-15);
}

TEST(GPUMatrix, to_sgmatrix)
{
	const int nrows = 3;
	const int ncols = 4;

	CGPUMatrix<float64_t> gpu_mat(nrows,ncols);
	for (int32_t i=0; i<nrows*ncols; i++)
		gpu_mat[i] = i;

	SGMatrix<float64_t> sg_mat = gpu_mat;

	for (int32_t i=0; i<nrows*ncols; i++)
		EXPECT_EQ(gpu_mat[i], sg_mat[i]);
}

TEST(GPUMatrix, from_sgmatrix)
{
	const int nrows = 3;
	const int ncols = 4;

	SGMatrix<float64_t> sg_mat(nrows,ncols);
	for (int32_t i=0; i<nrows*ncols; i++)
		sg_mat[i] = i;

	CGPUMatrix<float64_t> gpu_mat = sg_mat;

	for (int32_t i=0; i<nrows*ncols; i++)
		EXPECT_EQ(sg_mat[i], gpu_mat[i]);
}

TEST(GPUMatrix, to_eigen3)
{
	const int nrows = 3;
	const int ncols = 4;

	CGPUMatrix<float64_t> gpu_mat(nrows,ncols);
	for (int32_t i=0; i<nrows*ncols; i++)
		gpu_mat[i] = i;

	Eigen::MatrixXd eigen_mat = gpu_mat;

	for (int32_t i=0; i<nrows*ncols; i++)
		EXPECT_EQ(gpu_mat[i], eigen_mat(i));
}

TEST(GPUMatrix, from_eigen3)
{
	const int nrows = 3;
	const int ncols = 4;

	Eigen::MatrixXd eigen_mat(nrows,ncols);
	for (int32_t i=0; i<nrows*ncols; i++)
		eigen_mat(i) = i;

	CGPUMatrix<float64_t> gpu_mat = eigen_mat;

	for (int32_t i=0; i<nrows*ncols; i++)
		EXPECT_EQ(eigen_mat(i), gpu_mat[i]);
}

#endif // HAVE_CXX11
#endif // HAVE_VIENNACL
