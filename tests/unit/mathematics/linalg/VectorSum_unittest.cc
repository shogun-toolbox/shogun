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
#include <shogun/lib/SGVector.h>
#include <gtest/gtest.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUVector.h>
#endif

using namespace shogun;

#ifdef HAVE_EIGEN3
TEST(VectorSum, SGVector_explicit_eigen3_backend)
{
	const index_t size=10;
	SGVector<float64_t> a(size);
	a.set_const(1.0);

	float64_t result=linalg::vector_sum<linalg::Backend::EIGEN3>(a);

	EXPECT_NEAR(result, 10.0, 1E-15);
}

TEST(VectorSum, Eigen3_dynamic_explicit_eigen3_backend)
{
	index_t size=10;
	Eigen::VectorXd a=Eigen::VectorXd::Constant(size, 1);

	EXPECT_NEAR(linalg::vector_sum<linalg::Backend::EIGEN3>(a), 10.0, 1E-15);
}

TEST(VectorSum, Eigen3_fixed_explicit_eigen3_backend)
{
	Eigen::Vector3d a=Eigen::Vector3d::Constant(1);

	EXPECT_NEAR(linalg::vector_sum<linalg::Backend::EIGEN3>(a), 3.0, 1E-15);
}

#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL

TEST(VectorSum, viennacl_backend)
{
	const index_t size=10;
	CGPUVector<float64_t> a(size);
	a.set_const(1.0);

	float64_t result=linalg::vector_sum<linalg::Backend::VIENNACL>(a);

	EXPECT_NEAR(result, 10.0, 1E-15);
}

#endif // HAVE_VIENNACL

#endif // HAVE_LINALG_LIB
