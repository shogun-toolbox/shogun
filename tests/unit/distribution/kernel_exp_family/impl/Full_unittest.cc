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
#include <shogun/distributions/kernel_exp_family/impl/Full.h>
#include <shogun/distributions/kernel_exp_family/impl/kernel/Gaussian.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/some.h>
#include <shogun/base/init.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace kernel_exp_family_impl;

TEST(kernel_exp_family_impl_Full, compute_h_kernel_Gaussian)
{
	index_t N=3;
	index_t D=2;
	auto ND = N*D;
	SGMatrix<float64_t> X(D,N);
	X(0,0)=0;
	X(1,0)=1;
	X(0,1)=2;
	X(1,1)=4;
	X(0,2)=3;
	X(1,2)=6;
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda);
	
	auto result = est.compute_h();

	// from kernel_exp_family Python implementation
	float64_t reference[] = {0.00902188,  0.01353302,  0.01834103,  0.04119238,
							 -0.02736291,-0.0547254 };
	ASSERT_EQ(result.vlen, ND);
	for (auto i=0; i<ND; i++)
		EXPECT_NEAR(result[i], reference[i], 1e-8);
}

//TEST(kernel_exp_family_impl_Full, compute_xi_norm_2_kernel_Gaussian)
//{
//	index_t N=3;
//	index_t D=2;
//	SGMatrix<float64_t> X(D,N);
//	X(0,0)=0;
//	X(1,0)=1;
//	X(0,1)=2;
//	X(1,1)=4;
//	X(0,2)=3;
//	X(1,2)=6;
//
//	float64_t sigma = 2;
//	float64_t lambda = 1;
//	auto kernel = new kernel::Gaussian(sigma);
//	Full est(X, kernel, lambda);
//
//	auto result = est.compute_xi_norm_2();
//
//	// from kernel_exp_family Python implementation
//	EXPECT_NEAR(result, 2.5633762219921161, 1e-15);
//}

TEST(kernel_exp_family_impl_Full, fit_kernel_Gaussian)
{
	index_t N=3;
	index_t D=2;
	auto ND=N*D;
	SGMatrix<float64_t> X(D,N);
	X(0,0)=0;
	X(1,0)=1;
	X(0,1)=2;
	X(1,1)=4;
	X(0,2)=3;
	X(1,2)=6;
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda);
	
	est.fit();
	auto x = est.get_beta();
	ASSERT_EQ(x.vlen, ND);
	ASSERT(x.vector);
	
	// from kernel_exp_family Python implementation
	float64_t reference_x[] = {
			//-0.99999999999999989, // old alpha term, not used anymore, is -1.0/lambda
			0.00228091,  0.00342023,
		 0.00406425,  0.0092514 ,
		-0.00646103, -0.01294499};

	for (auto i=0; i<ND; i++)
		EXPECT_NEAR(x[i], reference_x[i], 1e-5);
	
}

TEST(kernel_exp_family_impl_Full, log_pdf_kernel_Gaussian)
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
	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda);
	est.fit();
	
	SGVector<float64_t> x(D);
	x[0] = 0;
	x[1] = 1;
	est.set_data(x);
	auto log_pdf = est.log_pdf(0);

	// from kernel_exp_family Python implementation
	EXPECT_NEAR(log_pdf, 0.6612075586873365, 1e-15);
}

TEST(kernel_exp_family_impl_Full, grad_kernel_Gaussian)
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
	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda);
	est.fit();
	
	SGVector<float64_t> x(D);
	x[0] = 0;
	x[1] = 1;
	est.set_data(x);
	auto grad = est.grad(0);

	// from kernel_exp_family Python implementation
	float64_t reference[] = {-0.01120102, -0.01680534};
	for (auto i=0; i<D; i++)
		EXPECT_NEAR(grad[i], reference[i], 1e-8);
	
	x[0] = 1;
	x[1] = 1;
	est.set_data(x);
	grad = est.grad(0);
	float64_t reference2[] = {-0.61982803, -0.04194253};
	for (auto i=0; i<D; i++)
		EXPECT_NEAR(grad[i], reference2[i], 1e-8);
}

TEST(kernel_exp_family_impl_Full, hessian_kernel_Gaussian)
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
	float64_t lambda = 2;
	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda);
	est.fit();
	
	SGVector<float64_t> x(D);
	x[0] = 1;
	x[1] = 1;
	est.set_data(x);
	auto hessian = est.hessian(0);
	
	float64_t reference[] = {0.20518773, -0.01275602,
							-0.01275602, -0.33620648};
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(hessian[i], reference[i], 1e-8);

	x[0] = -1;
	x[1] = 0;
	est.set_data(x);
	hessian = est.hessian(0);

	float64_t reference2[] = {0.12205638, 0.24511196,
							  0.24511196, 0.12173557};

	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(hessian[i], reference2[i], 1e-8);
}

TEST(kernel_exp_family_impl_Full, hessian_diag_kernel_Gaussian)
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
	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda);
	est.fit();
	
	SGVector<float64_t> x(D);
	x[0] = 0;
	x[1] = 1;
	est.set_data(x);
	auto hessian_diag = est.hessian_diag(0);
	
	// from kernel_exp_family Python implementation
	float64_t reference[] = {-1.34262346, -1.3600992 };
	
	for (auto i=0; i<D; i++)
		// TODO why is this only 1e-4? Check python code for that!
		EXPECT_NEAR(hessian_diag[i], reference[i], 1e-4);
}

TEST(kernel_exp_family_impl_Full, hessian_diag_equals_hessian)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();

	float64_t sigma = 2;
	float64_t lambda = 1;
	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda);
	est.fit();
	
	SGVector<float64_t> x(D);
	x[0] = CMath::randn_float();
	x[1] = CMath::randn_float();
	x[2] = CMath::randn_float();
	est.set_data(x);
	auto hessian = est.hessian(0);
	auto hessian_diag = est.hessian_diag(0);

	for (auto i=0; i<D; i++)
		EXPECT_NEAR(hessian_diag[i], hessian(i,i), 1e-8);
}

TEST(kernel_exp_family_impl_Full, score_kernel_Gaussian)
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
	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda);
	est.fit();
	
	// from kernel_exp_family Python implementation
	// on training data
	// TODO why is this only 1e-2? Check python code for that!
	EXPECT_NEAR(est.score(), -2.56402312081, 1e-2);
	
	// on test data
	SGVector<float64_t> x(D);
	x[0] = 0;
	x[1] = 1;
	est.set_data(x);
	// TODO why is this only 1e-2? Check python code for that!
	EXPECT_NEAR(est.score(), -2.70251871779, 1e-2);
}

TEST(kernel_exp_family_impl_Full, fit_base_measure_execute)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();

	float64_t sigma = 2;
	float64_t lambda = 1;
	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda, true);
	est.fit();
}

TEST(kernel_exp_family_impl_Full, log_pdf_base_measure_execute)
{
	get_global_parallel()->set_num_threads(1);
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();

	float64_t sigma = 2;
	float64_t lambda = 1;
	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda, true);
	est.fit();

	est.log_pdf();
}

TEST(kernel_exp_family_impl_Full, grad_base_measure_execute)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();

	float64_t sigma = 2;
	float64_t lambda = 1;
	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda, true);
	est.fit();

	est.grad();
}
