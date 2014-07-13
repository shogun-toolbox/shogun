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
#include <shogun/mathematics/linalg/linalg.h>
#include <shogun/lib/SGMatrix.h>
#include <gtest/gtest.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif // HAVE_EIGEN3

using namespace shogun;

#ifdef HAVE_EIGEN3
TEST(MatrixSum, SGMatrix_asymmetric_eigen3_backend_with_diag)
{
	const index_t m=2;
	const index_t n=3;
	SGMatrix<float64_t> mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	EXPECT_NEAR(linalg::sum<linalg::Backend::EIGEN3>(mat), 42.0, 1E-15);
}

TEST(MatrixSum, SGMatrix_symmetric_eigen3_backend_with_diag)
{
	const index_t n=3;
	SGMatrix<float64_t> mat(n, n);
	mat.set_const(1.0);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=i+1; j<n; ++j)
		{
			mat(i, j)=i*10+j+1;
			mat(j, i)=mat(i, j);
		}
	}

	EXPECT_NEAR(linalg::sum_symmetric<linalg::Backend::EIGEN3>(mat), 39.0, 1E-15);
}

TEST(MatrixSum, SGMatrix_asymmetric_block_eigen3_backend_with_diag)
{
	const index_t n=3;
	SGMatrix<float64_t> mat(n, n);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	float64_t sum=linalg::sum<linalg::Backend::EIGEN3>(linalg::block(mat,0,0,2,3));
	EXPECT_NEAR(sum, 42.0, 1E-15);
}

TEST(MatrixSum, SGMatrix_symmetric_block_eigen3_backend_with_diag)
{
	const index_t n=3;
	SGMatrix<float64_t> mat(n, n);
	mat.set_const(1.0);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=i+1; j<n; ++j)
		{
			mat(i, j)=i*10+j+1;
			mat(j, i)=mat(i, j);
		}
	}

	float64_t sum=linalg::sum<linalg::Backend::EIGEN3>(linalg::block(mat,1,1,2,2));
	EXPECT_NEAR(sum, 28.0, 1E-15);
}

TEST(MatrixSum, Eigen3_Matrix_asymmetric_eigen3_backend_with_diag)
{
	const index_t m=2;
	const index_t n=3;
	Eigen::MatrixXd mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	EXPECT_NEAR(linalg::sum<linalg::Backend::EIGEN3>(mat), 42.0, 1E-15);
}

TEST(MatrixSum, Eigen3_Matrix_symmetric_eigen3_backend_with_diag)
{
	const index_t n=3;
	Eigen::MatrixXd mat=Eigen::MatrixXd::Constant(n, n, 1);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=i+1; j<n; ++j)
		{
			mat(i, j)=i*10+j+1;
			mat(j, i)=mat(i, j);
		}
	}

	EXPECT_NEAR(linalg::sum_symmetric<linalg::Backend::EIGEN3>(mat), 39.0, 1E-15);
}

TEST(MatrixSum, Eigen3_Matrix_asymmetric_block_eigen3_backend_with_diag)
{
	const index_t n=3;
	Eigen::MatrixXd mat(n, n);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	float64_t sum=linalg::sum<linalg::Backend::EIGEN3>(linalg::block((SGMatrix<float64_t>)mat,0,0,2,3));
	EXPECT_NEAR(sum, 42.0, 1E-15);
}

TEST(MatrixSum, Eigen3_Matrix_symmetric_block_eigen3_backend_with_diag)
{
	const index_t n=3;
	Eigen::MatrixXd mat=Eigen::MatrixXd::Constant(n, n, 1);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=i+1; j<n; ++j)
		{
			mat(i, j)=i*10+j+1;
			mat(j, i)=mat(i, j);
		}
	}

	float64_t sum=linalg::sum<linalg::Backend::EIGEN3>(linalg::block((SGMatrix<float64_t>)mat,1,1,2,2));
	EXPECT_NEAR(sum, 28.0, 1E-15);
}

TEST(MatrixSum, SGMatrix_asymmetric_eigen3_backend_no_diag)
{
	const index_t m=2;
	const index_t n=3;
	SGMatrix<float64_t> mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	EXPECT_NEAR(linalg::sum<linalg::Backend::EIGEN3>(mat, true), 29.0, 1E-15);
}

TEST(MatrixSum, SGMatrix_symmetric_eigen3_backend_no_diag)
{
	const index_t n=3;
	SGMatrix<float64_t> mat(n, n);
	mat.set_const(1.0);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=i+1; j<n; ++j)
		{
			mat(i, j)=i*10+j+1;
			mat(j, i)=mat(i, j);
		}
	}

	EXPECT_NEAR(linalg::sum_symmetric<linalg::Backend::EIGEN3>(mat, true), 36.0, 1E-15);
}

TEST(MatrixSum, SGMatrix_asymmetric_block_eigen3_backend_no_diag)
{
	const index_t n=3;
	SGMatrix<float64_t> mat(n, n);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	float64_t sum=linalg::sum<linalg::Backend::EIGEN3>(linalg::block(mat,0,0,2,3),true);
	EXPECT_NEAR(sum, 29.0, 1E-15);
}

TEST(MatrixSum, SGMatrix_symmetric_block_eigen3_backend_no_diag)
{
	const index_t n=3;
	SGMatrix<float64_t> mat(n, n);
	mat.set_const(1.0);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=i+1; j<n; ++j)
		{
			mat(i, j)=i*10+j+1;
			mat(j, i)=mat(i, j);
		}
	}

	float64_t sum=linalg::sum<linalg::Backend::EIGEN3>(linalg::block(mat,1,1,2,2),true);
	EXPECT_NEAR(sum, 26.0, 1E-15);
}

TEST(MatrixSum, Eigen3_Matrix_asymmetric_eigen3_backend_no_diag)
{
	const index_t m=2;
	const index_t n=3;
	Eigen::MatrixXd mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	EXPECT_NEAR(linalg::sum<linalg::Backend::EIGEN3>(mat, true), 29.0, 1E-15);
}

TEST(MatrixSum, Eigen3_Matrix_symmetric_eigen3_backend_no_diag)
{
	const index_t n=3;
	Eigen::MatrixXd mat=Eigen::MatrixXd::Constant(n, n, 1);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=i+1; j<n; ++j)
		{
			mat(i, j)=i*10+j+1;
			mat(j, i)=mat(i, j);
		}
	}

	EXPECT_NEAR(linalg::sum_symmetric<linalg::Backend::EIGEN3>(mat, true), 36.0, 1E-15);
}

TEST(MatrixSum, Eigen3_Matrix_asymmetric_block_eigen3_backend_no_diag)
{
	const index_t n=3;
	Eigen::MatrixXd mat(n, n);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	float64_t sum=linalg::sum<linalg::Backend::EIGEN3>(linalg::block((SGMatrix<float64_t>)mat,0,0,2,3),true);
	EXPECT_NEAR(sum, 29.0, 1E-15);
}

TEST(MatrixSum, Eigen3_Matrix_symmetric_block_eigen3_backend_no_diag)
{
	const index_t n=3;
	Eigen::MatrixXd mat=Eigen::MatrixXd::Constant(n, n, 1);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=i+1; j<n; ++j)
		{
			mat(i, j)=i*10+j+1;
			mat(j, i)=mat(i, j);
		}
	}

	float64_t sum=linalg::sum<linalg::Backend::EIGEN3>(linalg::block((SGMatrix<float64_t>)mat,1,1,2,2),true);
	EXPECT_NEAR(sum, 26.0, 1E-15);
}

TEST(MatrixSum, SGMatrix_asymmetric_colwise_eigen3_backend_with_diag)
{
	const index_t m=2;
	const index_t n=3;
	SGMatrix<float64_t> mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::colwise_sum<linalg::Backend::EIGEN3>(mat);

	for (index_t j=0; j<n; ++j)
	{
		float64_t sum=0;
		for (index_t i=0; i<m; ++i)
			sum+=mat(i, j);
		EXPECT_NEAR(sum, s[j], 1E-15);
	}
}

TEST(MatrixSum, SGMatrix_symmetric_colwise_eigen3_backend_with_diag)
{
	const index_t n=3;
	SGMatrix<float64_t> mat(n, n);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::colwise_sum<linalg::Backend::EIGEN3>(mat);

	for (index_t j=0; j<n; ++j)
	{
		float64_t sum=0;
		for (index_t i=0; i<n; ++i)
			sum+=mat(i, j);
		EXPECT_NEAR(sum, s[j], 1E-15);
	}
}

TEST(MatrixSum, Eigen3_Matrix_asymmetric_colwise_eigen3_backend_with_diag)
{
	const index_t m=2;
	const index_t n=3;
	Eigen::MatrixXd mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::colwise_sum<linalg::Backend::EIGEN3>(mat);

	for (index_t j=0; j<n; ++j)
	{
		float64_t sum=0;
		for (index_t i=0; i<m; ++i)
			sum+=mat(i, j);
		EXPECT_NEAR(sum, s[j], 1E-15);
	}
}

TEST(MatrixSum, Eigen3_Matrix_symmetric_colwise_eigen3_backend_with_diag)
{
	const index_t n=3;
	Eigen::MatrixXd mat(n, n);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::colwise_sum<linalg::Backend::EIGEN3>(mat);

	for (index_t j=0; j<n; ++j)
	{
		float64_t sum=0;
		for (index_t i=0; i<n; ++i)
			sum+=mat(i, j);
		EXPECT_NEAR(sum, s[j], 1E-15);
	}
}

TEST(MatrixSum, SGMatrix_asymmetric_colwise_eigen3_backend_no_diag)
{
	const index_t m=2;
	const index_t n=3;
	SGMatrix<float64_t> mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::colwise_sum<linalg::Backend::EIGEN3>(mat, true);

	for (index_t j=0; j<n; ++j)
	{
		float64_t sum=0;
		for (index_t i=0; i<m; ++i)
			sum+=i==j ? 0 : mat(i, j);
		EXPECT_NEAR(sum, s[j], 1E-15);
	}
}

TEST(MatrixSum, SGMatrix_symmetric_colwise_eigen3_backend_no_diag)
{
	const index_t n=3;
	SGMatrix<float64_t> mat(n, n);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::colwise_sum<linalg::Backend::EIGEN3>(mat, true);

	for (index_t j=0; j<n; ++j)
	{
		float64_t sum=0;
		for (index_t i=0; i<n; ++i)
			sum+=i==j ? 0 : mat(i, j);
		EXPECT_NEAR(sum, s[j], 1E-15);
	}
}

TEST(MatrixSum, Eigen3_Matrix_asymmetric_colwise_eigen3_backend_no_diag)
{
	const index_t m=2;
	const index_t n=3;
	Eigen::MatrixXd mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::colwise_sum<linalg::Backend::EIGEN3>(mat, true);

	for (index_t j=0; j<n; ++j)
	{
		float64_t sum=0;
		for (index_t i=0; i<m; ++i)
			sum+=i==j ? 0 : mat(i, j);
		EXPECT_NEAR(sum, s[j], 1E-15);
	}
}

TEST(MatrixSum, Eigen3_Matrix_symmetric_colwise_eigen3_backend_no_diag)
{
	const index_t n=3;
	Eigen::MatrixXd mat(n, n);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::colwise_sum<linalg::Backend::EIGEN3>(mat, true);

	for (index_t j=0; j<n; ++j)
	{
		float64_t sum=0;
		for (index_t i=0; i<n; ++i)
			sum+=i==j ? 0 : mat(i, j);
		EXPECT_NEAR(sum, s[j], 1E-15);
	}
}

TEST(MatrixSum, SGMatrix_block_colwise_eigen3_backend_with_diag)
{
	const index_t m=2;
	const index_t n=3;
	SGMatrix<float64_t> mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::colwise_sum<linalg::Backend::EIGEN3>(
			linalg::block(mat, 0, 0, 2, 2));

	for (index_t j=0; j<2; ++j)
	{
		float64_t sum=0;
		for (index_t i=0; i<2; ++i)
			sum+=mat(i, j);
		EXPECT_NEAR(sum, s[j], 1E-15);
	}
}

TEST(MatrixSum, Eigen3_Matrix_block_colwise_eigen3_backend_with_diag)
{
	const index_t m=2;
	const index_t n=3;
	Eigen::MatrixXd mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::colwise_sum<linalg::Backend::EIGEN3>(
			linalg::block((SGMatrix<float64_t>)mat, 0, 0, 2, 2));

	for (index_t j=0; j<2; ++j)
	{
		float64_t sum=0;
		for (index_t i=0; i<2; ++i)
			sum+=mat(i, j);
		EXPECT_NEAR(sum, s[j], 1E-15);
	}
}

TEST(MatrixSum, SGMatrix_asymmetric_rowwise_eigen3_backend_with_diag)
{
	const index_t m=2;
	const index_t n=3;
	SGMatrix<float64_t> mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::rowwise_sum<linalg::Backend::EIGEN3>(mat);

	for (index_t i=0; i<m; ++i)
	{
		float64_t sum=0;
		for (index_t j=0; j<n; ++j)
			sum+=mat(i, j);
		EXPECT_NEAR(sum, s[i], 1E-15);
	}
}

TEST(MatrixSum, SGMatrix_symmetric_rowwise_eigen3_backend_with_diag)
{
	const index_t n=3;
	SGMatrix<float64_t> mat(n, n);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::rowwise_sum<linalg::Backend::EIGEN3>(mat);

	for (index_t i=0; i<n; ++i)
	{
		float64_t sum=0;
		for (index_t j=0; j<n; ++j)
			sum+=mat(i, j);
		EXPECT_NEAR(sum, s[i], 1E-15);
	}
}

TEST(MatrixSum, Eigen3_Matrix_asymmetric_rowwise_eigen3_backend_with_diag)
{
	const index_t m=2;
	const index_t n=3;
	Eigen::MatrixXd mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::rowwise_sum<linalg::Backend::EIGEN3>(mat);

	for (index_t i=0; i<m; ++i)
	{
		float64_t sum=0;
		for (index_t j=0; j<n; ++j)
			sum+=mat(i, j);
		EXPECT_NEAR(sum, s[i], 1E-15);
	}
}

TEST(MatrixSum, Eigen3_Matrix_symmetric_rowwise_eigen3_backend_with_diag)
{
	const index_t n=3;
	Eigen::MatrixXd mat(n, n);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::rowwise_sum<linalg::Backend::EIGEN3>(mat);

	for (index_t i=0; i<n; ++i)
	{
		float64_t sum=0;
		for (index_t j=0; j<n; ++j)
			sum+=mat(i, j);
		EXPECT_NEAR(sum, s[i], 1E-15);
	}
}

TEST(MatrixSum, SGMatrix_asymmetric_rowwise_eigen3_backend_no_diag)
{
	const index_t m=2;
	const index_t n=3;
	SGMatrix<float64_t> mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::rowwise_sum<linalg::Backend::EIGEN3>(mat, true);

	for (index_t i=0; i<m; ++i)
	{
		float64_t sum=0;
		for (index_t j=0; j<n; ++j)
			sum+=i==j ? 0 : mat(i, j);
		EXPECT_NEAR(sum, s[i], 1E-15);
	}
}

TEST(MatrixSum, SGMatrix_symmetric_rowwise_eigen3_backend_no_diag)
{
	const index_t n=3;
	SGMatrix<float64_t> mat(n, n);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::rowwise_sum<linalg::Backend::EIGEN3>(mat, true);

	for (index_t i=0; i<n; ++i)
	{
		float64_t sum=0;
		for (index_t j=0; j<n; ++j)
			sum+=i==j ? 0 : mat(i, j);
		EXPECT_NEAR(sum, s[i], 1E-15);
	}
}

TEST(MatrixSum, Eigen3_Matrix_asymmetric_rowwise_eigen3_backend_no_diag)
{
	const index_t m=2;
	const index_t n=3;
	Eigen::MatrixXd mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::rowwise_sum<linalg::Backend::EIGEN3>(mat, true);

	for (index_t i=0; i<m; ++i)
	{
		float64_t sum=0;
		for (index_t j=0; j<n; ++j)
			sum+=i==j ? 0 : mat(i, j);
		EXPECT_NEAR(sum, s[i], 1E-15);
	}
}

TEST(MatrixSum, Eigen3_Matrix_symmetric_rowwise_eigen3_backend_no_diag)
{
	const index_t n=3;
	Eigen::MatrixXd mat(n, n);

	for (index_t i=0; i<n; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::rowwise_sum<linalg::Backend::EIGEN3>(mat, true);

	for (index_t i=0; i<n; ++i)
	{
		float64_t sum=0;
		for (index_t j=0; j<n; ++j)
			sum+=i==j ? 0 : mat(i, j);
		EXPECT_NEAR(sum, s[i], 1E-15);
	}
}

TEST(MatrixSum, SGMatrix_block_rowwise_eigen3_backend_with_diag)
{
	const index_t m=2;
	const index_t n=3;
	SGMatrix<float64_t> mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::rowwise_sum<linalg::Backend::EIGEN3>(
			linalg::block(mat, 0, 0, 2, 2));

	for (index_t i=0; i<2; ++i)
	{
		float64_t sum=0;
		for (index_t j=0; j<2; ++j)
			sum+=mat(i, j);
		EXPECT_NEAR(sum, s[i], 1E-15);
	}
}

TEST(MatrixSum, Eigen3_Matrix_block_rowwise_eigen3_backend_with_diag)
{
	const index_t m=2;
	const index_t n=3;
	Eigen::MatrixXd mat(m, n);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<n; ++j)
			mat(i, j)=i*10+j+1;
	}

	SGVector<float64_t> s=linalg::rowwise_sum<linalg::Backend::EIGEN3>(
			linalg::block((SGMatrix<float64_t>)mat, 0, 0, 2, 2));

	for (index_t i=0; i<2; ++i)
	{
		float64_t sum=0;
		for (index_t j=0; j<2; ++j)
			sum+=mat(i, j);
		EXPECT_NEAR(sum, s[i], 1E-15);
	}
}
#endif // HAVE_EIGEN3

#endif // HAVE_LINALG_LIB
