/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Kunal Arora
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

TEST(RangeFillMatrix, native_backend)
{
	SGMatrix<index_t> m(2,2);

	linalg::range_fill<linalg::Backend::NATIVE>(m);

	for(int32_t i=0;i<4;i++)
	{
		EXPECT_EQ(m[i], i);
	}
}

TEST(RangeFillVector, native_backend)
{
	SGVector<float64_t> v(4);
	SGVector<index_t> v2(4);
	linalg::range_fill<linalg::Backend::NATIVE>(v,0.0);
	linalg::range_fill<linalg::Backend::NATIVE>(v2);
	EXPECT_EQ(v.vlen, 4);
	EXPECT_EQ(v2.vlen, 4);
	for(int32_t i=0;i<4;i++)
	{
		EXPECT_NEAR(v[i], i, 1e-9);
		EXPECT_NEAR(v2[i], i, 1e-9);
	}

}
TEST(RangeFillArray, native_backend)
{

	float64_t* w= SG_MALLOC(float64_t, 4);
	linalg::range_fill(w,4,0.0);
	for(int32_t i=0;i<4;i++)
	{
		EXPECT_NEAR(w[i], i, 1e-9);
	}

}


#endif // defined(HAVE_LINALG_LIB)
