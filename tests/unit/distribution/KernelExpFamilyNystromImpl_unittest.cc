/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Heiko Strathmann
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
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

#include <shogun/features/DenseFeatures.h>
#include <shogun/distributions/kernel_exp_family/impl/KernelExpFamilyImpl.h>
#include <shogun/distributions/kernel_exp_family/impl/KernelExpFamilyNystromImpl.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/some.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(KernelExpFamilyNystromImpl, kernel_dx_i)
{
	index_t N=3;
	index_t D=2;
	SGMatrix<float64_t> X(D,N);
	X(0,0)=0;
	X(1,0)=1;
	X(0,1)=2;
	X(1,1)=4;
	X(0,2)=3;
	X(1,2)=6;
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyNystromImpl est(X, sigma, lambda, N*D);
	
	index_t idx_a = 0;
	SGVector<float64_t> b(D);
	b[0]=-1;
	b[1]=3;
	auto result = est.kernel_dx(b, idx_a);
	
	// compare against full version
	for (auto i=0; i<D; i++)
	{
		auto entry = est.kernel_dx_i(b, idx_a, i);
		EXPECT_NEAR(result.vector[i], entry, 1e-15);
	}
}

TEST(KernelExpFamilyNystromImpl, kernel_dx_dx_i)
{
	index_t N=3;
	index_t D=2;
	SGMatrix<float64_t> X(D,N);
	X(0,0)=0;
	X(1,0)=1;
	X(0,1)=2;
	X(1,1)=4;
	X(0,2)=3;
	X(1,2)=6;
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyNystromImpl est(X, sigma, lambda, N*D);
	
	index_t idx_a = 0;
	SGVector<float64_t> b(D);
	b[0]=-1;
	b[1]=3;
	auto result = est.kernel_dx_dx(b, idx_a);
	
	// compare against full version
	for (auto i=0; i<D; i++)
	{
		auto entry = est.kernel_dx_dx_i(b, idx_a, i);
		EXPECT_NEAR(result.vector[i], entry, 1e-15);
	}
}

TEST(KernelExpFamilyNystromImpl, kernel_hessian_i_j)
{
	index_t N=30;
	index_t D=20;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyNystromImpl est(X, sigma, lambda, N*D);
	
	index_t idx_a = 0;
	index_t idx_b = 1;
	// compare against full version
	auto result = est.kernel_hessian(idx_a, idx_b);
	
	for (auto i=0; i<D; i++)
		for (auto j=0; j<D; j++)
		{
			auto entry = est.kernel_hessian_i_j(idx_a, idx_b, i, j);
			EXPECT_NEAR(result(i,j), entry, 1e-8);
		}
}

TEST(KernelExpFamilyNystromImpl, idx_to_ai)
{
	index_t D=3;
	KernelExpFamilyNystromImpl est(SGMatrix<float64_t>(D,1), 1, 1, 1);
	
	index_t idx=0;
	auto ai=est.idx_to_ai(idx);
	EXPECT_EQ(ai.first, 0);
	EXPECT_EQ(ai.second, 0);
	
	idx=1;
	ai=est.idx_to_ai(idx);
	EXPECT_EQ(ai.first, 0);
	EXPECT_EQ(ai.second, 1);
	
	idx=2;
	ai=est.idx_to_ai(idx);
	EXPECT_EQ(ai.first, 0);
	EXPECT_EQ(ai.second, 2);
	
	idx=3;
	ai=est.idx_to_ai(idx);
	EXPECT_EQ(ai.first, 1);
	EXPECT_EQ(ai.second, 0);
	
	idx=4;
	ai=est.idx_to_ai(idx);
	EXPECT_EQ(ai.first, 1);
	EXPECT_EQ(ai.second, 1);
}

TEST(KernelExpFamilyNystromImpl, compute_lower_right_submatrix_element)
{
	index_t N=5;
	index_t D=3;
	auto ND=N*D;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyNystromImpl est(X, sigma, lambda, N*D);
	
	auto result = est.build_system();
	auto A = result.first;

	// compare against full version
	for (auto row_idx=0; row_idx<ND; row_idx++)
		for (auto col_idx=0; col_idx<ND; col_idx++)
		{
			auto entry=est.compute_lower_right_submatrix_element(row_idx, col_idx);
			EXPECT_NEAR(entry, A(row_idx+1, col_idx+1), 1e-15);
		}
}

TEST(KernelExpFamilyNystromImpl, compute_first_row_no_storing)
{
	index_t N=5;
	index_t D=3;
	auto ND=N*D;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyNystromImpl est(X, sigma, lambda, N*D);
	
	auto result = est.build_system();
	auto A = result.first;
	auto first_row = est.compute_first_row_no_storing();
	ASSERT_EQ(first_row.vlen, ND);
	
	// compare against full version
	for (auto i=0; i<ND; i++)
		EXPECT_NEAR(first_row[i], A(0,i+1), 1e-15);
}

TEST(KernelExpFamilyNystromImpl, build_system_all_inds_equals_exact)
{
	index_t N=5;
	index_t D=3;
	auto ND=N*D;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyImpl est(X, sigma, lambda);
	KernelExpFamilyNystromImpl est_nystrom(X, sigma, lambda, ND);
	auto result = est.build_system();
	auto result_nystrom = est_nystrom.build_system();
	
	// compare against full version
	auto A = result.first;
	auto b = result.second;
	auto A_nystrom = result.first;
	auto b_nystrom = result.second;
	
	for (auto i=0; i<ND; i++)
		EXPECT_NEAR(b[i], b_nystrom[i], 1e-15);
	
	for (auto i=0; i<ND*ND; i++)
		EXPECT_NEAR(A.matrix[i], A_nystrom.matrix[i], 1e-15);
}

TEST(KernelExpFamilyNystromImpl, fit_all_inds_equals_exact)
{
	index_t N=5;
	index_t D=3;
	auto ND=N*D;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyImpl est(X, sigma, lambda);
	KernelExpFamilyNystromImpl est_nystrom(X, sigma, lambda, ND);
	
	est.fit();
	est_nystrom.fit();
	
	// compare against full version
	auto result_nystrom=est_nystrom.get_alpha_beta();
	auto result=est.get_alpha_beta();
	
	ASSERT_EQ(result.vlen, ND+1);
	ASSERT_EQ(result_nystrom.vlen, ND+1);
	ASSERT(result.vector);
	ASSERT(result_nystrom.vector);
	
	
	for (auto i=0; i<ND+1; i++)
		EXPECT_NEAR(result[i], result_nystrom[i], 1e-12);
}

TEST(KernelExpFamilyNystromImpl, fit_half_inds_shape)
{
	index_t N=5;
	index_t D=3;
	auto ND=N*D;
	index_t m=ND/2;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyNystromImpl est(X, sigma, lambda, m);
	est.fit();
	
	auto alpha_beta=est.get_alpha_beta();
	ASSERT_EQ(alpha_beta.vlen, m+1);
	ASSERT(alpha_beta.vector);
}

TEST(KernelExpFamilyNystromImpl, pinv_non_square1)
{
	index_t N=3;
	index_t D=2;
	SGMatrix<float64_t> X(D,N);
	X(0,0)=0;
	X(1,0)=1;
	X(0,1)=2;
	X(1,1)=4;
	X(0,2)=3;
	X(1,2)=6;
		
	auto pinv = KernelExpFamilyNystromImpl::pinv(X);
	
	ASSERT_EQ(pinv.num_rows, N);
	ASSERT_EQ(pinv.num_cols, D);

	// from numpy.linalg.pinv
	// using rcond=np.finfo(np.float32).eps * np.max((N,D))=3.57627868652e-07
	float64_t reference[] = {-2.00000000e+00,   1.53846154e-01,   2.30769231e-01,
         1.00000000e+00,   7.63278329e-17,  -4.16333634e-17};
	
	for (auto i=0; i<pinv.num_rows*pinv.num_cols; i++)
		EXPECT_NEAR(pinv[i], reference[i], 1e-9);
}

TEST(KernelExpFamilyNystromImpl, pinv_square)
{
	index_t N=2;
	index_t D=2;
	SGMatrix<float64_t> X(D,N);
	X(0,0)=0;
	X(1,0)=1;
	X(0,1)=2;
	X(1,1)=4;
		
	auto pinv = KernelExpFamilyNystromImpl::pinv(X);
	
	ASSERT_EQ(pinv.num_rows, 2);
	ASSERT_EQ(pinv.num_cols, 2);

	// from numpy.linalg.pinv
	float64_t reference[] = {-2.00000000e+00, 5.00000000e-01, 
							 1.00000000e+00,  -2.77555756e-17};
	
	for (auto i=0; i<pinv.num_rows*pinv.num_cols; i++)
		EXPECT_NEAR(pinv[i], reference[i], 1e-15);
}

TEST(KernelExpFamilyNystromImpl, log_pdf_all_inds_equals_exact)
{
	index_t N=3;
	index_t D=2;
	SGMatrix<float64_t> X(D,N);
	X(0,0)=0;
	X(1,0)=1;
	X(0,1)=2;
	X(1,1)=4;
	X(0,2)=3;
	X(1,2)=6;
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	auto m=N*D;
	KernelExpFamilyNystromImpl est_nystrom(X, sigma, lambda, m);
	KernelExpFamilyImpl est(X, sigma, lambda);
	est_nystrom.fit();
	est.fit();
	
	SGVector<float64_t> x(D);
	x[0] = 0;
	x[1] = 1;
	auto log_pdf = est.log_pdf(x);
	auto log_pdf_nystrom = est_nystrom.log_pdf(x);
	
	EXPECT_NEAR(log_pdf, log_pdf_nystrom, 1e-8);
}

TEST(KernelExpFamilyNystromImpl, log_pdf_half_inds_execute)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	auto m = N*D;
	KernelExpFamilyNystromImpl est(X, sigma, lambda, m);
	est.fit();
	
	SGVector<float64_t> x(D);
	x[0] = 0;
	x[1] = 1;
	x[2] = 2;

	est.log_pdf(x);
}
