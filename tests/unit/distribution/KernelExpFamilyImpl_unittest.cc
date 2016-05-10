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
#include <shogun/distributions/KernelExpFamilyImpl.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/some.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(KernelExpFamilyImpl, kernel_equals_manual)
{
	index_t N=3;
	index_t D=2;
	SGMatrix<float64_t> X(D,N);
	X(0,0)=0;
	X(1,0)=2;
	X(0,1)=4;
	X(1,1)=8;
	X(0,2)=3;
	X(1,2)=6;
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyImpl est(X, sigma, lambda);
	
	for (auto idx_a=0; idx_a<N; idx_a++)
	{
		for (auto idx_b=0; idx_b<N; idx_b++)
		{
			float64_t k = 0;
			for (auto i=0; i<D; i++)
			{
				float64_t d = X(i,idx_a)-X(i,idx_b);
				k+=d*d;
			}
			k = exp(-k/sigma);
			EXPECT_NEAR(k, est.kernel(idx_a, idx_b), 1e-15);
		}
	}
}

TEST(KernelExpFamilyImpl, kernel_equals_shogun)
{
	index_t N=30;
	index_t D=20;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyImpl est(X, sigma, lambda);
	
	auto k = new CGaussianKernel();
	SG_REF(k);
	auto f = new CDenseFeatures<float64_t>(X);
	SG_REF(f);
	k->set_width(sigma);
	k->init(f,f);
	
	for (auto idx_a=0; idx_a<N; idx_a++)
	{
		for (auto idx_b=0; idx_b<N; idx_b++)
			EXPECT_NEAR(k->kernel(idx_a, idx_b), est.kernel(idx_a, idx_b), 1e-15);
	}
	SG_UNREF(k);
	SG_UNREF(f);
}
TEST(KernelExpFamilyImpl, kernel_dx_dx_dy)
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
	KernelExpFamilyImpl est(X, sigma, lambda);
	
	index_t idx_a = 0;
	index_t idx_b = 1;
	auto result = est.kernel_dx_dx_dy(idx_a, idx_b);
	
	// from kernel_exp_family Python implementation
	float64_t reference[] = {-0.00300688, -0.02405503,
							 -0.01353095, -0.02706191};
	ASSERT_EQ(result.num_rows, D);
	ASSERT_EQ(result.num_cols, D);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(result.matrix[i], reference[i], 1e-8);
	
	idx_a = 0;
	idx_b = 0;
	result = est.kernel_dx_dx_dy(idx_a, idx_b);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(result.matrix[i], 0, 1e-8);
	
	idx_a = 1;
	idx_b = 1;
	result = est.kernel_dx_dx_dy(idx_a, idx_b);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(result.matrix[i], 0, 1e-8);
}

TEST(KernelExpFamilyImpl, kernel_dx_dx_dy_dy)
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
	KernelExpFamilyImpl est(X, sigma, lambda);
	index_t idx_a = 0;
	index_t idx_b = 1;
	auto result = est.kernel_dx_dx_dy_dy(idx_a, idx_b);
	
	// from kernel_exp_family Python implementation
	float64_t reference[] = {-0.0075172, 0.03608254,
							 0.03608254 , 0.04510318};
	ASSERT_EQ(result.num_rows, D);
	ASSERT_EQ(result.num_cols, D);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(result.matrix[i], reference[i], 1e-8);
		
	idx_a = 0;
	idx_b = 0;
	result = est.kernel_dx_dx_dy_dy(idx_a, idx_b);
	float64_t reference2[] = { 3.,  1., 1.,  3.};
	ASSERT_EQ(result.num_rows, D);
	ASSERT_EQ(result.num_cols, D);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(result.matrix[i], reference2[i], 1e-8);
	
}

TEST(KernelExpFamilyImpl, kernel_dx_dx)
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
	KernelExpFamilyImpl est(X, sigma, lambda);
	
	index_t idx_a = 0;
	index_t idx_b = 1;
	auto result = est.kernel_dx_dx(idx_a, idx_b);
	
	// from kernel_exp_family Python implementation
	float64_t reference[] = {-0.00451032, -0.00902064,
							 -0.00902064 , -0.01202751};
	ASSERT_EQ(result.num_rows, D);
	ASSERT_EQ(result.num_cols, D);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(result.matrix[i], reference[i], 1e-8);
		
	idx_a = 0;
	idx_b = 0;
	result = est.kernel_dx_dx(idx_a, idx_b);
	ASSERT_EQ(result.num_rows, D);
	ASSERT_EQ(result.num_cols, D);
	for (auto i=0; i<D; i++)
		for (auto j=0; i<D; i++)
		{
			float64_t ref;
			if (i==j)
				ref=1;
			else
				ref=0;
				
			EXPECT_EQ(result(i,j), ref);
		}
}

TEST(KernelExpFamilyImpl, kernel_dx_dx_i_j)
{
	index_t N=30;
	index_t D=20;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyImpl est(X, sigma, lambda);
	
	index_t idx_a = 0;
	index_t idx_b = 1;
	// compare against full version
	auto result = est.kernel_dx_dx(idx_a, idx_b);
	
	for (auto i=0; i<D; i++)
		for (auto j=0; j<D; j++)
		{
			auto entry = est.kernel_dx_dx_i_j(idx_a, idx_b, i, j);
			EXPECT_NEAR(result(i,j), entry, 1e-8);
		}
	
}

TEST(KernelExpFamilyImpl, kernel_dx_dx_all)
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
	KernelExpFamilyImpl est(X, sigma, lambda);
	
	auto result = est.kernel_dx_dx_all();
	
	// from kernel_exp_family Python implementation
	float64_t reference[] = {
		1.0 ,0.0 ,-0.00451031757893 ,-0.00902063515787 ,-3.31195017503e-07 ,-6.20990657818e-07 ,
		0.0 ,1.0 ,-0.00902063515787 ,-0.0120275135438 ,-6.20990657818e-07 ,-9.93585052508e-07 ,
		-0.00451031757893 ,-0.00902063515787 ,1.0 ,0.0 ,0.0 ,-0.164169997248 ,-0.00902063515787 ,
		-0.0120275135438 ,0.0 ,1.0 ,-0.164169997248 ,-0.246254995872 ,-3.31195017503e-07 ,
		-6.20990657818e-07 ,0.0 ,-0.164169997248 ,1.0 ,0.0 ,-6.20990657818e-07 ,-9.93585052508e-07 ,
		-0.164169997248 ,-0.246254995872 ,0.0 ,1.0 	};
							 
	ASSERT_EQ(result.num_rows, ND);
	ASSERT_EQ(result.num_cols, ND);
	for (auto i=0; i<ND*ND; i++)
		EXPECT_NEAR(result.matrix[i], reference[i], 1e-8);
}

TEST(KernelExpFamilyImpl, compute_h)
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
	KernelExpFamilyImpl est(X, sigma, lambda);
	
	auto result = est.compute_h();

	// from kernel_exp_family Python implementation
	float64_t reference[] = {0.00902188,  0.01353302,  0.01834103,  0.04119238,
							 -0.02736291,-0.0547254 };
	ASSERT_EQ(result.vlen, ND);
	for (auto i=0; i<ND; i++)
		EXPECT_NEAR(result[i], reference[i], 1e-8);
}

TEST(KernelExpFamilyImpl, compute_xi_norm_2)
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
	KernelExpFamilyImpl est(X, sigma, lambda);
	
	auto result = est.compute_xi_norm_2();
	
	// from kernel_exp_family Python implementation
	EXPECT_NEAR(result, 2.5633762219921161, 1e-15);
}

TEST(KernelExpFamilyImpl, build_system)
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
	KernelExpFamilyImpl est(X, sigma, lambda);
	
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

TEST(KernelExpFamilyImpl, fit)
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
	KernelExpFamilyImpl est(X, sigma, lambda);
	
	auto x = est.fit();
	ASSERT_EQ(x.vlen, ND+1);
	
	// from kernel_exp_family Python implementation
	float64_t reference_x[] = {-0.99999999999999989, 0.00228091,  0.00342023,
         0.00406425,  0.0092514 ,
        -0.00646103, -0.01294499};

	for (auto i=0; i<ND+1; i++)
		EXPECT_NEAR(x[i], reference_x[i], 1e-5);
	
}

TEST(KernelExpFamilyImpl, idx_to_ai)
{
	index_t D=3;
	KernelExpFamilyImpl est(SGMatrix<float64_t>(D,1), 1, 1);
	
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

TEST(KernelExpFamilyImpl, compute_lower_right_submatrix_element)
{
	index_t N=10;
	index_t D=5;
	auto ND=N*D;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyImpl est(X, sigma, lambda);
	
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

TEST(KernelExpFamilyImpl, compute_first_row_no_storing)
{
	index_t N=10;
	index_t D=5;
	auto ND=N*D;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyImpl est(X, sigma, lambda);
	
	auto result = est.build_system();
	auto A = result.first;
	auto first_row = est.compute_first_row_no_storing();
	ASSERT_EQ(first_row.vlen, ND);
	
	// compare against full version
	for (auto i=0; i<ND; i++)
		EXPECT_NEAR(first_row[i], A(0,i+1), 1e-15);
}

TEST(KernelExpFamilyImpl, build_system_nystrom_all_inds_equals_exact)
{
	index_t N=10;
	index_t D=5;
	auto ND=N*D;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyImpl est(X, sigma, lambda);
	
	auto result = est.build_system();
	auto A = result.first;
	auto b = result.second;
	
	SGVector<index_t> inds(ND);
	inds.range_fill();
	
	// compare against full version
	auto result_nystrom = est.build_system_nystrom(inds);
	auto A_nystrom = result.first;
	auto b_nystrom = result.second;
	
	for (auto i=0; i<ND; i++)
		EXPECT_NEAR(b[i], b_nystrom[i], 1e-15);
	
	for (auto i=0; i<ND*ND; i++)
		EXPECT_NEAR(A.matrix[i], A_nystrom.matrix[i], 1e-15);
	
}

TEST(KernelExpFamilyImpl, fit_nystrom_all_inds_equals_exact)
{
	index_t N=10;
	index_t D=5;
	auto ND=N*D;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyImpl est(X, sigma, lambda);
	
	auto result = est.fit();
	
	SGVector<index_t> inds(ND);
	inds.range_fill();
	
	// compare against full version
	auto result_nystrom = est.fit_nystrom(inds);
	
	ASSERT_EQ(result.vlen, ND+1);
	ASSERT_EQ(result_nystrom.vlen, ND+1);
	
	for (auto i=0; i<ND+1; i++)
		EXPECT_NEAR(result[i], result_nystrom[i], 1e-12);
}

TEST(KernelExpFamilyImpl, pinv_non_square1)
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
	KernelExpFamilyImpl est(X, sigma, lambda);
	
	auto pinv = est.pinv(X);
	
	ASSERT_EQ(pinv.num_rows, N);
	ASSERT_EQ(pinv.num_cols, D);

	// from numpy.linalg.pinv
	// using rcond=np.finfo(np.float32).eps * np.max((N,D))=3.57627868652e-07
	float64_t reference[] = {-2.00000000e+00,   1.53846154e-01,   2.30769231e-01,
         1.00000000e+00,   7.63278329e-17,  -4.16333634e-17};
	
	for (auto i=0; i<pinv.num_rows*pinv.num_cols; i++)
		EXPECT_NEAR(pinv[i], reference[i], 1e-9);
}

TEST(KernelExpFamilyImpl, pinv_square)
{
	index_t N=2;
	index_t D=2;
	SGMatrix<float64_t> X(D,N);
	X(0,0)=0;
	X(1,0)=1;
	X(0,1)=2;
	X(1,1)=4;
		
	float64_t sigma = 2;
	float64_t lambda = 1;
	KernelExpFamilyImpl est(X, sigma, lambda);
	
	auto pinv = est.pinv(X);
	
	ASSERT_EQ(pinv.num_rows, 2);
	ASSERT_EQ(pinv.num_cols, 2);

	// from numpy.linalg.pinv
	float64_t reference[] = {-2.00000000e+00, 5.00000000e-01, 
							 1.00000000e+00,  -2.77555756e-17};
	
	for (auto i=0; i<pinv.num_rows*pinv.num_cols; i++)
		EXPECT_NEAR(pinv[i], reference[i], 1e-15);
}
