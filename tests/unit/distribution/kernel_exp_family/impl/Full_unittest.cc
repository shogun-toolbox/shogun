/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016-2017 Heiko Strathmann, Dougal Sutherland
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

/* All unit tests are based on the following gist
 * https://gist.github.com/karlnapf/c0b24fc18d946cc315733ed679e249e8
 */

const float64_t sigma = 2.0;
const float64_t lambda = 1.0;
const index_t N=3;
const index_t D=2;
const index_t ND=N*D;

SGMatrix<float64_t> data_train_fixed()
{
	SGMatrix<float64_t> X(D,N);
	X(0,0)=0;
	X(1,0)=1;
	X(0,1)=2;
	X(1,1)=4;
	X(0,2)=3;
	X(1,2)=6;

	return X;
}

SGMatrix<float64_t> data_test_fixed()
{
	SGMatrix<float64_t> X(D,2);
	X(0,0)=0;
	X(1,0)=1;
	X(0,1)=1;
	X(1,1)=1;

	return X;
}

SGMatrix<float64_t> data_train_random()
{
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<ND; i++)
		X.matrix[i] = CMath::randn_float();

	return X;
}

TEST(kernel_exp_family_impl_Full, compute_h_kernel_Gaussian)
{
	auto X = data_train_fixed();
		
	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda);
	
	auto result = est.compute_h();

	float64_t reference[] = {0.00902188,  0.01353302,  0.01834103,  0.04119238,
							 -0.02736291,-0.0547254 };
	ASSERT_EQ(result.vlen, ND);
	for (auto i=0; i<ND; i++)
		EXPECT_NEAR(result[i], reference[i], 1e-8);
}

TEST(kernel_exp_family_impl_Full, fit_kernel_Gaussian)
{
	auto X = data_train_fixed();
		
	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda);
	
	est.fit();
	auto x = est.get_beta();
	ASSERT_EQ(x.vlen, ND);
	ASSERT(x.vector);
	
	float64_t reference_x[] = {
			0.00228091,  0.00342023,  0.00406425,  0.0092514 , -0.00646103,
			-0.01294499
	};

	for (auto i=0; i<ND; i++)
		EXPECT_NEAR(x[i], reference_x[i], 1e-8);
	
}

TEST(kernel_exp_family_impl_Full, log_pdf_kernel_Gaussian)
{
	auto X = data_train_fixed();

	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda);
	est.fit();
	
	SGVector<float64_t> x(D);
	x[0] = 0;
	x[1] = 1;
	est.set_data(x);
	auto log_pdf = est.log_pdf(0);

	EXPECT_NEAR(log_pdf, 0.6610996707107812, 1e-15);
}

TEST(kernel_exp_family_impl_Full, grad_kernel_Gaussian)
{
	auto X = data_train_fixed();

	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda);
	est.fit();
	
	SGVector<float64_t> x(D);
	x[0] = 0;
	x[1] = 1;
	est.set_data(x);
	auto grad = est.grad(0);

	float64_t reference[] = {-0.00684274, -0.0102607 };
	for (auto i=0; i<D; i++)
		EXPECT_NEAR(grad[i], reference[i], 1e-8);
	
	x[0] = 1;
	x[1] = 1;
	est.set_data(x);
	grad = est.grad(0);
	float64_t reference2[] = {-0.62020189, -0.03895487};
	for (auto i=0; i<D; i++)
		EXPECT_NEAR(grad[i], reference2[i], 1e-8);
}

TEST(kernel_exp_family_impl_Full, hessian_kernel_Gaussian)
{
	auto X = data_train_fixed();

	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda);
	est.fit();
	
	SGVector<float64_t> x(D);
	x[0] = 0;
	x[1] = 1;
	est.set_data(x);
	auto hessian = est.hessian(0);
	
	float64_t reference[] = {-1.34299555, -0.02133143,
							-0.02133143, -1.36075253};
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(hessian[i], reference[i], 1e-8);

	x[0] = 1;
	x[1] = 1;
	est.set_data(x);
	hessian = est.hessian(0);

	float64_t reference2[] = { 0.40612246, -0.02956325,
								-0.02956325, -0.67672628};

	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(hessian[i], reference2[i], 1e-8);
}

TEST(kernel_exp_family_impl_Full, hessian_diag_equals_hessian)
{
	auto X = data_train_random();

	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda);
	est.fit();
	
	// on training data
	for (auto i=0; i<N; i++)
	{
		auto hessian = est.hessian(i);
		auto hessian_diag = est.hessian_diag(i);

		for (auto j=0; j<D; j++)
			EXPECT_NEAR(hessian_diag[j], hessian(j,j), 1e-8);
	}

	auto X_test = data_train_random();
	est.set_data(X_test);
	for (auto i=0; i<N; i++)
	{
		auto hessian = est.hessian(i);
		auto hessian_diag = est.hessian_diag(i);

		for (auto j=0; j<D; j++)
			EXPECT_NEAR(hessian_diag[j], hessian(j,j), 1e-8);
	}
}

TEST(kernel_exp_family_impl_Full, score_kernel_Gaussian)
{
	auto X = data_train_fixed();

	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda);
	est.fit();
	
	EXPECT_NEAR(est.score(), -2.56147602838, 1e-8);
	
	auto X_test = data_test_fixed();
	est.set_data(X_test);
	EXPECT_NEAR(est.score(), -1.39059595876, 1e-8);
}

TEST(kernel_exp_family_impl_Full, fit_base_measure_execute)
{
	auto X = data_train_random();

	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda, true);
	est.fit();
}

TEST(kernel_exp_family_impl_Full, log_pdf_base_measure_execute)
{
	auto X = data_train_random();

	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda, true);
	est.fit();

	est.log_pdf();
}

TEST(kernel_exp_family_impl_Full, grad_base_measure_execute)
{
	auto X = data_train_random();

	auto kernel = new kernel::Gaussian(sigma);
	Full est(X, kernel, lambda, true);
	est.fit();

	est.grad();
}
