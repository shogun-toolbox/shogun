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

#include <shogun/lib/GPUVector.h>
#include <viennacl/linalg/inner_prod.hpp>
#include <gtest/gtest.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif

using namespace shogun;

TEST(GPUVector, element_read_write)
{
	const int n = 9;
	
	CGPUVector<float64_t> vec(n);
	
	for (int32_t i=0; i<n; i++)
		vec[i] = i;
	
	for (int32_t i=0; i<n; i++)
		EXPECT_EQ(i, vec[i]);
}

TEST(GPUVector, zero)
{
	const int n = 9;
	
	CGPUVector<float64_t> vec(n);
	vec.zero();
	
	for (int32_t i=0; i<n; i++)
		EXPECT_EQ(0, vec[i]);
}

/** Tests element access operations on a vector that was created with an offset */
TEST(GPUVector, element_access_with_offset)
{
	CGPUVector<float64_t> data(25);
	for (int32_t i=0; i<25; i++)
		data[i] = i;
	
	CGPUVector<float64_t> vec(data.vector, 9, 7);
	
	for (int32_t i=0; i<9; i++)
		EXPECT_EQ(data[i+7], vec[i]);
}

/** Tests dot product performed using ViennaCL on two CGPUVector objects
 * that were created with offsets
 */
TEST(GPUVector, dot_product_with_offset)
{
	CGPUVector<float64_t> data(24);
	for (int32_t i=0; i<24; i++)
		data[i] = i;
	
	CGPUVector<float64_t> A(data.vector, 12, 0);
	CGPUVector<float64_t> B(data.vector, 12, 12);
	
	float c = viennacl::linalg::inner_prod(A.vcl_vector(), B.vcl_vector());
	
	float c_sg = SGVector<float64_t>::dot(
		((SGVector<float64_t>)A).vector, ((SGVector<float64_t>)B).vector, 12);
	
	EXPECT_NEAR(c_sg, c, 1e-15);
}

TEST(GPUVector, to_sgvector)
{
	const int n = 9;
	
	CGPUVector<float64_t> gpu_vec(9);
	for (int32_t i=0; i<n; i++)
		gpu_vec[i] = i;
	
	SGVector<float64_t> sg_vec = gpu_vec;
	
	for (int32_t i=0; i<n; i++)
		EXPECT_EQ(gpu_vec[i], sg_vec[i]);
}

TEST(GPUVector, from_sgvector)
{
	const int n = 9;
	
	SGVector<float64_t> sg_vec(9);
	for (int32_t i=0; i<n; i++)
		sg_vec[i] = i;
	
	CGPUVector<float64_t> gpu_vec = sg_vec;
	
	for (int32_t i=0; i<n; i++)
		EXPECT_EQ(sg_vec[i], gpu_vec[i]);
}

#ifdef HAVE_EIGEN3

TEST(GPUVector, to_eigen3_column_vector)
{
	const int n = 9;
	
	CGPUVector<float64_t> gpu_vec(9);
	for (int32_t i=0; i<n; i++)
		gpu_vec[i] = i;
	
	Eigen::VectorXd eigen_vec = gpu_vec;
	
	for (int32_t i=0; i<n; i++)
		EXPECT_EQ(gpu_vec[i], eigen_vec[i]);
}

TEST(GPUVector, from_eigen3_column_vector)
{
	const int n = 9;
	
	Eigen::VectorXd eigen_vec(9);
	for (int32_t i=0; i<n; i++)
		eigen_vec[i] = i;
	
	CGPUVector<float64_t> gpu_vec = eigen_vec;
	
	for (int32_t i=0; i<n; i++)
		EXPECT_EQ(eigen_vec[i], gpu_vec[i]);
}

TEST(GPUVector, to_eigen3_row_vector)
{
	const int n = 9;
	
	CGPUVector<float64_t> gpu_vec(9);
	for (int32_t i=0; i<n; i++)
		gpu_vec[i] = i;
	
	Eigen::RowVectorXd eigen_vec = gpu_vec;
	
	for (int32_t i=0; i<n; i++)
		EXPECT_EQ(gpu_vec[i], eigen_vec[i]);
}

TEST(GPUVector, from_eigen3_row_vector)
{
	const int n = 9;
	
	Eigen::RowVectorXd eigen_vec(9);
	for (int32_t i=0; i<n; i++)
		eigen_vec[i] = i;
	
	CGPUVector<float64_t> gpu_vec = eigen_vec;
	
	for (int32_t i=0; i<n; i++)
		EXPECT_EQ(eigen_vec[i], gpu_vec[i]);
}

#endif

#endif
