/*
 * Copyright (c) 2014, Shogun Toolbox Foundation
 * All rights reserved.
 *
 * Written (W) 2014 Sunil K. Mahendrakar
 * Written (W) 2014 Soumyajit De
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
 */

#include <shogun/lib/config.h>

#ifdef HAVE_LINALG_LIB
#include <shogun/mathematics/linalg/linalg.h>
#include <shogun/lib/SGVector.h>
#include <algorithm>
#include <gtest/gtest.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUVector.h>
#endif // HAVE_VIENNACL

using namespace shogun;

TEST(DotProduct, SGVector_default_backend)
{
	const index_t size=10;
	SGVector<float64_t> a(size), b(size);
	a.set_const(1.0);
	b.set_const(2.0);

	EXPECT_NEAR(linalg::dot(a, b), 20.0, 1E-15);
}

#ifdef HAVE_EIGEN3
TEST(DotProduct, SGVector_explicit_eigen3_backend)
{
	const index_t size=10;
	SGVector<float64_t> a(size), b(size);
	a.set_const(1.0);
	b.set_const(2.0);

	float64_t result=linalg::dot<linalg::Backend::EIGEN3>(a, b);

	EXPECT_NEAR(result, 20.0, 1E-15);
}

TEST(DotProduct, Eigen3_dynamic_default_backend)
{
	index_t size=10;
	Eigen::VectorXd a=Eigen::VectorXd::Constant(size, 1);
	Eigen::VectorXd b=Eigen::VectorXd::Constant(size, 2);

	EXPECT_NEAR(linalg::dot(a, b), 20.0, 1E-15);
}

TEST(DotProduct, Eigen3_fixed_default_backend)
{
	Eigen::Vector3d a=Eigen::Vector3d::Constant(1);
	Eigen::Vector3d b=Eigen::Vector3d::Constant(2);

	EXPECT_NEAR(linalg::dot(a, b), 6.0, 1E-15);
}

TEST(DotProduct, Eigen3_dynamic_explicit_eigen3_backend)
{
	index_t size=10;
	Eigen::VectorXd a=Eigen::VectorXd::Constant(size, 1);
	Eigen::VectorXd b=Eigen::VectorXd::Constant(size, 2);

	EXPECT_NEAR(linalg::dot<linalg::Backend::EIGEN3>(a, b), 20.0, 1E-15);
}

TEST(DotProduct, Eigen3_fixed_explicit_eigen3_backend)
{
	Eigen::Vector3d a=Eigen::Vector3d::Constant(1);
	Eigen::Vector3d b=Eigen::Vector3d::Constant(2);

	EXPECT_NEAR(linalg::dot<linalg::Backend::EIGEN3>(a, b), 6.0, 1E-15);
}

#ifdef HAVE_VIENNACL
TEST(DotProduct, Eigen3_dynamic_explicit_viennacl_backend)
{
	index_t size=10;
	Eigen::VectorXf a=Eigen::VectorXf::Constant(size, 1);
	Eigen::VectorXf b=Eigen::VectorXf::Constant(size, 2);

	EXPECT_NEAR(linalg::dot<linalg::Backend::VIENNACL>(a, b), 20.0, 1E-6);
}

TEST(DotProduct, Eigen3_fixed_explicit_viennacl_backend)
{
	Eigen::Vector3f a=Eigen::Vector3f::Constant(1);
	Eigen::Vector3f b=Eigen::Vector3f::Constant(2);

	EXPECT_NEAR(linalg::dot<linalg::Backend::VIENNACL>(a, b), 6.0, 1E-6);
}
#endif // HAVE_VIENNACL
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
TEST(DotProduct, SGVector_explicit_viennacl_backend)
{
	const index_t size=10;
	SGVector<float32_t> a(size), b(size);
	a.set_const(1.0);
	b.set_const(2.0);

	float64_t result=linalg::dot<linalg::Backend::VIENNACL>(a, b);

	EXPECT_NEAR(result, 20.0, 1E-6);
}

TEST(DotProduct, ViennaCL_default_backend)
{
	const index_t size=10;
	CGPUVector<float32_t> a(size), b(size);
	a.set_const(1.0);
	b.set_const(2.0);

	EXPECT_NEAR(linalg::dot(a, b), 20.0, 1E-15);
}

TEST(DotProduct, ViennaCL_explicit_viennacl_backend)
{
	const index_t size=10;
	CGPUVector<float32_t> a(size), b(size);
	a.set_const(1.0);
	b.set_const(2.0);

	float32_t result=linalg::dot<linalg::Backend::VIENNACL>(a, b);

	EXPECT_NEAR(result, 20.0, 1E-15);
}

#ifdef HAVE_EIGEN3
TEST(DotProduct, ViennaCL_explicit_eigen3_backend)
{
	const index_t size=10;
	CGPUVector<float32_t> a(size), b(size);
	a.set_const(1.0);
	b.set_const(2.0);

	float32_t result=linalg::dot<linalg::Backend::EIGEN3>(a, b);

	EXPECT_NEAR(result, 20.0, 1E-15);
}
#endif // HAVE_EIGEN3
#endif // HAVE_VIENNACL

#endif // HAVE_LINALG_LIB
