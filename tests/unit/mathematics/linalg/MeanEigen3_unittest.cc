/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Pan Deng
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
#include <shogun/mathematics/linalg/linalg.h>
#include <shogun/lib/SGVector.h>
#include <gtest/gtest.h>

#include <shogun/mathematics/eigen3.h>

using namespace shogun;

#ifdef HAVE_LINALG_LIB

TEST(MeanEigen3, SGVector_explicit_eigen3_backend_float64)
{
	const index_t size = 10;
	SGVector<float64_t> a(size);
	for (index_t i = 0; i < 10; ++i) a[i] = i;
	float64_t result = linalg::mean<linalg::Backend::EIGEN3>(a);
	EXPECT_NEAR(result, 4.5, 1E-15);
}

TEST(MeanEigen3, SGVector_explicit_eigen3_backend_int32)
{
	index_t size = 10;
	SGVector<int32_t> a(size);
	for (index_t i = 0; i < 10; ++i) a[i] = i;
	float64_t result = linalg::mean<linalg::Backend::EIGEN3>(a);
	EXPECT_NEAR(result, 4.5, 1E-15);
}

TEST(MeanEigen3, SGVector_explicit_eigen3_backend_int64)
{
	index_t size = 10;
	SGVector<int64_t> a(size);
	for (index_t i = 0; i < 10; ++i) a[i] = i;
	float64_t result = linalg::mean<linalg::Backend::EIGEN3>(a);
	EXPECT_NEAR(result, 4.5, 1E-15);
}

TEST(MeanEigen3, Eigen3_dynamic_explicit_eigen3_backend_float)
{
	const index_t size = 10;
	Eigen::VectorXd a(size);
	a << 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
	EXPECT_NEAR(linalg::mean<linalg::Backend::EIGEN3>(a), 4.5, 1E-15);
}

TEST(MeanEigen3, Eigen3_dynamic_explicit_eigen3_backend_int)
{
	index_t size = 10;
	Eigen::VectorXd a(size); 
	a << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
	EXPECT_NEAR(linalg::mean<linalg::Backend::EIGEN3>(a), 4.5, 1E-15);
}

#endif // HAVE_LINALG_LIB

