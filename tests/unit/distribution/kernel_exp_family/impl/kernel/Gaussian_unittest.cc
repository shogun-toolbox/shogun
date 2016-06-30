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
#include <shogun/distributions/kernel_exp_family/impl/kernel/Gaussian.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/some.h>
#include <gtest/gtest.h>
#include <memory>

using namespace std;
using namespace shogun;
using namespace shogun::kernel_exp_family_impl::kernel;

TEST(kernel_exp_family_impl_kernel_Gaussian, kernel_equals_manual)
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
	auto kernel = make_shared<Gaussian>(sigma);
	kernel->set_lhs(X);
	kernel->set_rhs(X);
	
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

			EXPECT_NEAR(k, kernel->kernel(idx_a, idx_b), 1e-15);
		}
	}
}

TEST(kernel_exp_family_impl_kernel_Gaussian, kernel_equals_shogun)
{
	index_t N=30;
	index_t D=20;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
    auto kernel = make_shared<Gaussian>(sigma);
	kernel->set_lhs(X);
	kernel->set_rhs(X);
	
	auto k = new CGaussianKernel();
	SG_REF(k);
	auto f = new CDenseFeatures<float64_t>(X);
	SG_REF(f);
	k->set_width(sigma);
	k->init(f,f);
	
	for (auto idx_a=0; idx_a<N; idx_a++)
	{
		for (auto idx_b=0; idx_b<N; idx_b++)
			EXPECT_NEAR(k->kernel(idx_a, idx_b), kernel->kernel(idx_a, idx_b), 1e-15);
	}
	SG_UNREF(k);
	SG_UNREF(f);
}

TEST(kernel_exp_family_impl_kernel_Gaussian, dx)
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
    auto kernel = make_shared<Gaussian>(sigma);
	kernel->set_lhs(X);
	
	index_t idx_a = 0;
	SGVector<float64_t> b(D);
	b[0]=-1;
	b[1]=3;
	kernel->set_rhs(b);
	auto result = kernel->dx(idx_a, 0);
	
	// from kernel_exp_family Python implementation
	float64_t reference[] = {-0.082085, 0.16417 };
	ASSERT_EQ(result.vlen, D);
	for (auto i=0; i<D; i++)
		EXPECT_NEAR(result.vector[i], reference[i], 1e-8);
	
	idx_a = 1;
	result = kernel->dx(idx_a, 0);
	float64_t reference2[] = {-0.02021384,  -0.00673795};
	for (auto i=0; i<D; i++)
		EXPECT_NEAR(result.vector[i], reference2[i], 1e-8);
}

TEST(kernel_exp_family_impl_kernel_Gaussian, dx_dx_dy)
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
    auto kernel = make_shared<Gaussian>(sigma);
	kernel->set_lhs(X);
	kernel->set_rhs(X);
	
	index_t idx_a = 0;
	index_t idx_b = 1;
	auto result = kernel->dx_dx_dy(idx_a, idx_b);
	
	// from kernel_exp_family Python implementation
	float64_t reference[] = {-0.00300688, -0.02405503,
							 -0.01353095, -0.02706191};
	ASSERT_EQ(result.num_rows, D);
	ASSERT_EQ(result.num_cols, D);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(result.matrix[i], reference[i], 1e-8);
	
	idx_a = 0;
	idx_b = 0;
	result = kernel->dx_dx_dy(idx_a, idx_b);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(result.matrix[i], 0, 1e-8);
	
	idx_a = 1;
	idx_b = 1;
	result = kernel->dx_dx_dy(idx_a, idx_b);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(result.matrix[i], 0, 1e-8);
}

TEST(kernel_exp_family_impl_kernel_Gaussian, dx_dx_dy_dy)
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
    auto kernel = make_shared<Gaussian>(sigma);
	kernel->set_lhs(X);
	kernel->set_rhs(X);
	index_t idx_a = 0;
	index_t idx_b = 1;
	auto result = kernel->dx_dx_dy_dy(idx_a, idx_b);
	
	// from kernel_exp_family Python implementation
	float64_t reference[] = {-0.0075172, 0.03608254,
							 0.03608254 , 0.04510318};
	ASSERT_EQ(result.num_rows, D);
	ASSERT_EQ(result.num_cols, D);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(result.matrix[i], reference[i], 1e-8);
		
	idx_a = 0;
	idx_b = 0;
	result = kernel->dx_dx_dy_dy(idx_a, idx_b);
	float64_t reference2[] = { 3.,  1., 1.,  3.};
	ASSERT_EQ(result.num_rows, D);
	ASSERT_EQ(result.num_cols, D);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(result.matrix[i], reference2[i], 1e-8);
}

TEST(kernel_exp_family_impl_kernel_Gaussian, dx_dy)
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
    auto kernel = make_shared<Gaussian>(sigma);
	kernel->set_lhs(X);
	kernel->set_rhs(X);
	
	index_t idx_a = 0;
	index_t idx_b = 1;
	auto result = kernel->dx_dy(idx_a, idx_b);
	
	// from kernel_exp_family Python implementation
	float64_t reference[] = {-0.00451032, -0.00902064,
							 -0.00902064 , -0.01202751};
	ASSERT_EQ(result.num_rows, D);
	ASSERT_EQ(result.num_cols, D);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(result.matrix[i], reference[i], 1e-8);
		
	idx_a = 0;
	idx_b = 0;
	result = kernel->dx_dy(idx_a, idx_b);
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

TEST(kernel_exp_family_impl_kernel_Gaussian, dx_dx)
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
    auto kernel = make_shared<Gaussian>(sigma);
	kernel->set_lhs(X);
	
	index_t idx_a=0;
	index_t idx_b=1;
	SGVector<float64_t> a(X.get_column_vector(idx_a), D, false);
	kernel->set_rhs(a);
	
	auto result = kernel->dx_dx(idx_b, idx_a);
	float64_t reference[] = { 0.00451032,  0.01202751};
	
	ASSERT_EQ(result.vlen, D);
	for (auto i=0; i<D; i++)
		EXPECT_NEAR(result.vector[i], reference[i], 1e-8);
}

TEST(kernel_exp_family_impl_kernel_Gaussian, dx_i_dx_i_dx_j)
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
    auto kernel = make_shared<Gaussian>(sigma);
	kernel->set_lhs(X);
	
	index_t idx_a=0;
	index_t idx_b=1;
	SGVector<float64_t> a(X.get_column_vector(idx_a), D, false);
	
	kernel->set_rhs(a);
	auto result = kernel->dx_i_dx_i_dx_j(idx_b, 0);
	
	// from kernel_exp_family Python implementation
	float64_t reference[] = { -0.00300688,  -0.02405503, -0.01353095,  -0.02706191};
	ASSERT_EQ(result.num_rows, D);
	ASSERT_EQ(result.num_cols, D);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(result.matrix[i], reference[i], 1e-8);
	
	idx_b=2;
	result = kernel->dx_i_dx_i_dx_j(idx_b, 0);
	float64_t reference2[] = {-7.45188789e-07,   -2.98075516e-06, -1.65597509e-06,  -4.55393149e-06};
	ASSERT_EQ(result.num_rows, D);
	ASSERT_EQ(result.num_cols, D);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(result.matrix[i], reference2[i], 1e-8);
}

TEST(kernel_exp_family_impl_kernel_Gaussian, kernel_dx_i_dx_j)
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
    auto kernel = make_shared<Gaussian>(sigma);
	kernel->set_lhs(X);
	
	index_t idx_a=0;
	index_t idx_b=1;
	SGVector<float64_t> a(X.get_column_vector(idx_a), D, false);
	
	kernel->set_rhs(a);
	auto result = kernel->dx_i_dx_j(idx_b, 0);
	
	// from kernel_exp_family Python implementation
	float64_t reference[] = { 0.00451032,  0.00902064, 0.00902064,  0.01202751};
	ASSERT_EQ(result.num_rows, D);
	ASSERT_EQ(result.num_cols, D);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(result.matrix[i], reference[i], 1e-8);
	
	idx_b=2;
	result = kernel->dx_i_dx_j(idx_b, 0);
	float64_t reference2[] = { 3.31195018e-07,   6.20990658e-07,  6.20990658e-07,   9.93585053e-07};
	ASSERT_EQ(result.num_rows, D);
	ASSERT_EQ(result.num_cols, D);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(result.matrix[i], reference2[i], 1e-8);
}

TEST(kernel_exp_family_impl_kernel_Gaussian, dx_dy_all)
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
    auto kernel = make_shared<Gaussian>(sigma);
	kernel->set_lhs(X);
	kernel->set_rhs(X);
	
	auto result = kernel->dx_dy_all();
	
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

TEST(kernel_exp_family_impl_kernel_Gaussian, dx_dx_dy_dy_sum)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;
    auto kernel = make_shared<Gaussian>(sigma);
	kernel->set_lhs(X);
	kernel->set_rhs(X);
	
	// compare against batch version
	auto idx_a=0;
	auto idx_b=1;
	auto mat = kernel->dx_dx_dy_dy(idx_a, idx_b);
	float64_t sum_manual=0;
	for (auto i=0; i<D*D; i++)
		sum_manual += mat.matrix[i];
		
	auto sum = kernel->dx_dx_dy_dy_sum(idx_a, idx_b);
	EXPECT_NEAR(sum, sum_manual, 1e-14);
}

TEST(kernel_exp_family_impl_kernel_Gaussian, dx_component)
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

    auto kernel = make_shared<Gaussian>(sigma);
	kernel->set_lhs(X);
	
	index_t idx_a = 0;
	SGVector<float64_t> b(D);
	b[0]=-1;
	b[1]=3;
	kernel->set_rhs(b);
	auto result = kernel->dx(idx_a, 0);
	
	// compare against full version
	for (auto i=0; i<D; i++)
	{
		auto entry = kernel->dx_component(idx_a, 0, i);
		EXPECT_NEAR(result.vector[i], entry, 1e-15);
	}
}

TEST(kernel_exp_family_impl_kernel_Gaussian, dx_dx_component)
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

    auto kernel = make_shared<Gaussian>(sigma);
	kernel->set_lhs(X);
	
	index_t idx_a = 0;
	SGVector<float64_t> b(D);
	b[0]=-1;
	b[1]=3;
	kernel->set_rhs(b);
	auto result = kernel->dx_dx(idx_a, 0);
	
	// compare against full version
	for (auto i=0; i<D; i++)
	{
		auto entry = kernel->dx_dx_component(idx_a, 0, i);
		EXPECT_NEAR(result.vector[i], entry, 1e-15);
	}
}

TEST(kernel_exp_family_impl_kernel_Gaussian, dx_i_dx_j_component)
{
	index_t N=30;
	index_t D=20;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;

    auto kernel = make_shared<Gaussian>(sigma);
	kernel->set_lhs(X);
	
	index_t idx_a = 0;
	index_t idx_b = 1;
	SGVector<float64_t> a(X.get_column_vector(idx_a), D, false);
	// compare against full version
	kernel->set_rhs(a);
	auto result = kernel->dx_i_dx_j(idx_b, idx_a);
	
	for (auto i=0; i<D; i++)
	{
		auto entry = kernel->dx_i_dx_j_component(idx_b, idx_a, i);
		for (auto j=0; j<D; j++)
			EXPECT_NEAR(result(i,j), entry[j], 1e-8);
	}
}

TEST(kernel_exp_family_impl_kernel_Gaussian, dx_i_dx_i_dx_j_component)
{
	index_t N=30;
	index_t D=20;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;

    auto kernel = make_shared<Gaussian>(sigma);
	kernel->set_lhs(X);
	
	index_t idx_a = 0;
	index_t idx_b = 1;
	SGVector<float64_t> a(X.get_column_vector(idx_a), D, false);
	// compare against full version
	kernel->set_rhs(a);
	auto result = kernel->dx_i_dx_i_dx_j(idx_b, idx_a);
	
	for (auto i=0; i<D; i++)
	{
		auto entry = kernel->dx_i_dx_i_dx_j_component(idx_b, idx_a, i);
		for (auto j=0; j<D; j++)
			EXPECT_NEAR(result(i,j), entry[j], 1e-8);
	}
}

TEST(kernel_exp_family_impl_kernel_Gaussian, dx_dy_component)
{
	index_t N=30;
	index_t D=20;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();
		
	float64_t sigma = 2;

    auto kernel = make_shared<Gaussian>(sigma);
	kernel->set_lhs(X);
	kernel->set_rhs(X);
	
	index_t idx_a = 0;
	index_t idx_b = 1;
	// compare against full version
	auto result = kernel->dx_dy(idx_a, idx_b);
	
	for (auto i=0; i<D; i++)
		for (auto j=0; j<D; j++)
		{
			auto entry = kernel->dx_dy_component(idx_a, idx_b, i, j);
			EXPECT_NEAR(result(i,j), entry, 1e-8);
		}
}
