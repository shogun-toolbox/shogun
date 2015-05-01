/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Khaled Nasr
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

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUMatrix.h>
#include <shogun/lib/GPUVector.h>
#endif

using namespace shogun;

#ifdef HAVE_EIGEN3
TEST(SetRowsConst, eigen3_backend)
{
	SGMatrix<float64_t> A(3,4);
	SGVector<float64_t> v(A.num_rows);

	for (int32_t i=0; i<v.vlen; i++)
		v[i] = i;

	linalg::set_rows_const<linalg::Backend::EIGEN3>(A, v);

	for (int32_t i=0; i<A.num_rows; i++)
		for (int32_t j=0; j<A.num_cols; j++)
			EXPECT_EQ(A(i,j), v[i]);
}
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
TEST(SetRowsConst, viennacl_backend)
{
	CGPUMatrix<float64_t> A(3,4);
	CGPUVector<float64_t> v(A.num_rows);

	for (int32_t i=0; i<v.vlen; i++)
		v[i] = i;

	linalg::set_rows_const<linalg::Backend::VIENNACL>(A, v);

	for (int32_t i=0; i<A.num_rows; i++)
		for (int32_t j=0; j<A.num_cols; j++)
			EXPECT_EQ(A(i,j), v[i]);
}
#endif // HAVE_VIENNACL

#endif // HAVE_LINALG_LIB
