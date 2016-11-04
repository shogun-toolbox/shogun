/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Kunal Arora
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
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <gtest/gtest.h>

using namespace shogun;

//test when start=0.0
TEST(RangeFillVector, eigen3_backend_start0)
{
	SGVector<float64_t> v(4);
	linalg::range_fill<linalg::Backend::EIGEN3>(v,0.0);
	EXPECT_EQ(v.vlen, 4);
	for(int32_t i=0;i<4;i++)
		EXPECT_NEAR(v[i], i, 1e-9);
}

//test when start not given as input
TEST(RangeFillVector, eigen3_backend_no_start)
{
	SGVector<float64_t> v(4);
	linalg::range_fill<linalg::Backend::EIGEN3>(v);
	EXPECT_EQ(v.vlen, 4);
	for(int32_t i=0;i<4;i++)
		EXPECT_NEAR(v[i], i, 1e-9);
}
//test when start>0
TEST(RangeFillVector, eigen3_backend_start_greaterThan_0)
{
	SGVector<float64_t> v(4);
	linalg::range_fill<linalg::Backend::EIGEN3>(v,5.0);
	EXPECT_EQ(v.vlen, 4);
	for(int32_t i=0;i<4;i++)
		EXPECT_NEAR(v[i], i+5.0, 1e-9);
}

//test for negative values
TEST(RangeFillVector, eigen3_backend_start_lessThan_0)
{
	SGVector<float64_t> v(4);
	linalg::range_fill<linalg::Backend::EIGEN3>(v,-5.0);
	EXPECT_EQ(v.vlen, 4);
	for(int32_t i=0;i<4;i++)
		EXPECT_NEAR(v[i], i-5.0, 1e-9);
}

#endif // defined(HAVE_LINALG_LIB)
