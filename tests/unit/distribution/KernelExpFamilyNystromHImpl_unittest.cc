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
#include <shogun/distributions/kernel_exp_family/impl/KernelExpFamilyNystromHImpl.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/base/some.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(KernelExpFamilyNystromHImpl, kernel_dx_dx_dy_dy_component)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyNystromHImpl est(X, sigma, lambda, N*D);
	
	auto result = est.kernel_dx_dx_dy_dy(0, 1);
	
	// compare against full version
	for (auto i=0; i<D; i++)
	{
	    for (auto j=0; j<D; j++)
	    {
		    auto entry = est.kernel_dx_dx_dy_dy_component(0, 1, i, j);
		    EXPECT_NEAR(result(i,j), entry, 1e-15);
	    }
	}
}

TEST(KernelExpFamilyNystromHImpl, compute_xi_norm_2_all_inds_equals_exact)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyNystromHImpl est(X, sigma, lambda, N*D);
	KernelExpFamilyImpl est_full(X, sigma, lambda);
	
	// compare against full version
    EXPECT_NEAR(est.compute_xi_norm_2(), est_full.compute_xi_norm_2(), 1e-12);
}

TEST(KernelExpFamilyNystromHImpl, compute_h_all_inds_equals_full)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyNystromHImpl est(X, sigma, lambda, N*D);
	KernelExpFamilyImpl est_full(X, sigma, lambda);
	
	// compare against full version
	auto h = est.compute_h();
	auto h_full = est.compute_h();
	
	ASSERT_EQ(h.vlen, h_full.vlen);
	
	for (auto i=0; i<N*D; i++)
        EXPECT_NEAR(h[i], h_full[i], 1e-12);
}

TEST(KernelExpFamilyNystromHImpl, compute_h_half_inds_equals_subsampled_full)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	
	index_t m=5;
	SGVector<index_t> temp(N*D);
	temp.range_fill();
	CMath::permute(temp);
	SGVector<index_t> inds(m);
	memcpy(inds.vector, temp.vector, sizeof(index_t)*m);
	
	KernelExpFamilyNystromHImpl est(X, sigma, lambda, inds);
	KernelExpFamilyImpl est_full(X, sigma, lambda);
	
	// compare against full version
	auto h = est.compute_h();
	auto h_full = est_full.compute_h();
	
	ASSERT_EQ(h.vlen, m);
	ASSERT_EQ(h_full.vlen, N*D);
	for (auto i=0; i<m; i++)
        EXPECT_NEAR(h[i], h_full[inds[i]], 1e-12);
}

TEST(KernelExpFamilyNystromHImpl, build_system_equals_build_system_from_full)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyNystromHImpl est(X, sigma, lambda, N*D);
	
	// compare against full version
	auto result=est.build_system();
	auto result_full=est.build_system_from_full();
	
	ASSERT_EQ(result.first.num_rows, result_full.first.num_rows);
	ASSERT_EQ(result.first.num_cols, result_full.first.num_cols);
	ASSERT_EQ(result.second.vlen, result_full.second.vlen);

	for (auto i=0; i<N*D*N*D; i++)
		EXPECT_NEAR(result.first.matrix[i], result_full.first.matrix[i], 1e-8);
		
	for (auto i=0; i<N*D; i++)
		EXPECT_NEAR(result.second[i], result_full.second[i], 1e-8);
}

TEST(KernelExpFamilyNystromHImpl, fit_all_inds_equals_exact)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyNystromHImpl est(X, sigma, lambda, N*D);
	KernelExpFamilyImpl est_full(X, sigma, lambda);
	
	est.fit();
	est_full.fit();
	
	// compare against full version
	auto result=est.get_alpha_beta();
	auto result_full=est_full.get_alpha_beta();
	
	EXPECT_EQ(result.vlen, result_full.vlen);

	for (auto i=0; i<N*D; i++)
		EXPECT_NEAR(result[i], result_full[i], 1e-8);
}

TEST(KernelExpFamilyNystromHImpl, fit_half_inds_shape)
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
	KernelExpFamilyNystromHImpl est(X, sigma, lambda, m);
	est.fit();
	
	auto alpha_beta=est.get_alpha_beta();
	ASSERT_EQ(alpha_beta.vlen, m+1);
	ASSERT(alpha_beta.vector);
}

TEST(KernelExpFamilyNystromHImpl, fit)
{
	index_t N=5;
	index_t D=3;
	index_t m=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyNystromHImpl est(X, sigma, lambda, m);
	est.fit();
	
	auto alpha_beta=est.get_alpha_beta();
	ASSERT_EQ(alpha_beta.vlen, m+1);
	ASSERT(alpha_beta.vector);
}

TEST(KernelExpFamilyNystromHImpl, log_pdf_almost_all_inds_close_exact)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	auto m=N*D-1;
	KernelExpFamilyNystromHImpl est_nystrom(X, sigma, lambda, m);
	KernelExpFamilyImpl est(X, sigma, lambda);
	est_nystrom.fit();
	est.fit();
	
	SGVector<float64_t> x(D);
	for (auto i=0; i<D; i++)
		x[i]=CMath::randn_float();

	auto log_pdf = est.log_pdf(x);
	auto log_pdf_nystrom = est_nystrom.log_pdf(x);
	
	EXPECT_NEAR(log_pdf, log_pdf_nystrom, 0.3);
}

TEST(KernelExpFamilyNystromHImpl, grad_all_inds_equals_exact)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	auto m=N*D;
	KernelExpFamilyNystromHImpl est_nystrom(X, sigma, lambda, m);
	KernelExpFamilyImpl est(X, sigma, lambda);
	est_nystrom.fit();
	est.fit();
	
	SGVector<float64_t> x(D);
	for (auto i=0; i<D; i++)
		x[i]=CMath::randn_float();

	auto grad = est.grad(x);
	auto grad_nystrom = est_nystrom.grad(x);
	
	for (auto i=0; i<D; i++)
		EXPECT_NEAR(grad[i], grad_nystrom[i], 1e-8);
}

TEST(KernelExpFamilyNystromHImpl, grad_almost_all_inds_close_exact)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	auto m=N*D-1;
	KernelExpFamilyNystromHImpl est_nystrom(X, sigma, lambda, m);
	KernelExpFamilyImpl est(X, sigma, lambda);
	est_nystrom.fit();
	est.fit();
	
	SGVector<float64_t> x(D);
	for (auto i=0; i<D; i++)
		x[i]=CMath::randn_float();

	auto grad = est.grad(x);
	auto grad_nystrom = est_nystrom.grad(x);
	
	for (auto i=0; i<D; i++)
		EXPECT_NEAR(grad[i], grad_nystrom[i], 0.3);
}

TEST(KernelExpFamilyNystromHImpl, kernel_dx_dx_dy_component)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	
	KernelExpFamilyNystromHImpl est(X, sigma, lambda, N*D);
	
	// compare against full version
	for (auto idx_a=0; idx_a<N; idx_a++)
	    for (auto idx_b=0; idx_b<N; idx_b++)
	    {
	        auto result = est.kernel_dx_dx_dy(idx_a,idx_b);
	
	        ASSERT_EQ(result.num_rows, D);
	        ASSERT_EQ(result.num_cols, D);
	        for (auto i=0; i<D; i++)
	            for (auto j=0; j<D; j++)
	            {
	                auto comp = est.kernel_dx_dx_dy_component(idx_a,idx_b, i, j);
                    EXPECT_NEAR(result(i,j), comp, 1e-12);
                }
        }
}
