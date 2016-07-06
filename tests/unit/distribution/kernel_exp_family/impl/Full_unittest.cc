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

TEST(kernel_exp_family_impl_Full, compute_xi_norm_2_kernel_Gaussian)
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
	
	auto result = est.compute_xi_norm_2();
	
	// from kernel_exp_family Python implementation
	EXPECT_NEAR(result, 2.5633762219921161, 1e-15);
}

TEST(kernel_exp_family_impl_Full, build_system_kernel_Gaussian)
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
	
	auto result = est.build_system();
	auto A = result.first;
	auto b = result.second;
	
	auto ND=N*D;
	ASSERT_EQ(A.num_rows, ND+1);
	ASSERT_EQ(A.num_cols, ND+1);
	ASSERT_EQ(b.vlen, ND+1);
	
	// from kernel_exp_family Python implementation
	float64_t reference_A[] = {
2.56539000231 ,0.0118777487667 ,0.0178237575117 ,0.0273952084559 ,0.0608313131137 ,
-0.0387380656692 ,-0.0773521682976 ,0.0118777487667 ,1.33336723827 ,4.97272472278e-05 ,
-0.00751716198221 ,-0.0150343228317 ,0.000493087224651 ,0.000986243448264 ,
0.0178237575117 ,4.97272472278e-05 ,1.33340867765 ,-0.0150343375575 ,-0.0200457403653 ,
0.000657150637366 ,0.00147926167395 ,0.0273952084559 ,-0.00751716198221 ,
-0.0150343375575 ,1.34235116761 ,0.0135256212451 ,2.3651749567e-09 ,-0.273616658158 ,
0.0608313131137 ,-0.0150343228317 ,-0.0200457403653 ,0.0135256212451 ,1.36260644798 ,
-0.273616658594 ,-0.410424987269 ,-0.0387380656692 ,0.000493087224651 ,0.000657150637366 ,
2.3651749567e-09 ,-0.273616658594 ,1.34231726267 ,0.0134758939984 ,-0.0773521682976 ,
0.000986243448264 ,0.00147926167395 ,-0.273616658158 ,-0.410424987269 ,0.0134758939984 ,
1.36253110366};


	float64_t reference_b[] = { -2.56337622, -0.00902188, -0.01353302, -0.01834103, -0.04119238,
								0.02736291,  0.0547254 };

	for (auto i=0; i<(ND+1)*(ND+1); i++)
		EXPECT_NEAR(A.matrix[i], reference_A[i], 1e-8);
	
	for (auto i=0; i<ND+1; i++)
		EXPECT_NEAR(b[i], reference_b[i], 1e-8);
	
}

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
	auto x = est.get_alpha_beta();
	ASSERT_EQ(x.vlen, ND+1);
	ASSERT(x.vector);
	
	// from kernel_exp_family Python implementation
	float64_t reference_x[] = {-0.99999999999999989, 0.00228091,  0.00342023,
		 0.00406425,  0.0092514 ,
		-0.00646103, -0.01294499};

	for (auto i=0; i<ND+1; i++)
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
	auto log_pdf = est.log_pdf(x);

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
	auto grad = est.grad(x);

	// from kernel_exp_family Python implementation
	float64_t reference[] = {-0.01120102, -0.01680534};
	for (auto i=0; i<D; i++)
		EXPECT_NEAR(grad[i], reference[i], 1e-8);
	
	x[0] = 1;
	x[1] = 1;
	grad = est.grad(x);
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
	auto hessian = est.hessian(x);
	
	float64_t reference[] = {0.20518773, -0.01275602,
							-0.01275602, -0.33620648};
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(hessian[i], reference[i], 1e-8);

	x[0] = -1;
	x[1] = 0;

	hessian = est.hessian(x);

	float64_t reference2[] = {0.12205638, 0.24511196,
							  0.24511196, 0.12173557};

	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(hessian[i], reference2[i], 1e-8);
}
