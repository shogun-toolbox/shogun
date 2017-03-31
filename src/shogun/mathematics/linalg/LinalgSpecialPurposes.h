/*
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: 2017 Pan Deng, 2014 Khaled Nasr
 */

#ifndef LINALG_SPECIAL_PURPOSE_H_
#define LINALG_SPECIAL_PURPOSE_H_

#include <shogun/mathematics/linalg/LinalgNamespace.h>

namespace shogun
{

namespace linalg
{

/** Applies the elementwise logistic function f(x) = 1/(1+exp(-x)) to a matrix
 *  This method returns the result in-place.
 *
 * @param a The input matrix
 * @param result The output matrix
 */
template <typename T>
void logistic(SGMatrix<T>& a, SGMatrix<T>& result)
{
	REQUIRE((a.num_rows == result.num_rows),
		"Number of rows of matrix a (%d) must match matrix result (%d).\n",
		a.num_rows, result.num_rows);
	REQUIRE((a.num_cols == result.num_cols),
		"Number of columns of matrix result (%d) must match matrix result (%d).\n",
		a.num_cols, result.num_cols);

	infer_backend(a, result)->logistic(a, result);
}

}

}

#endif //LINALG_SPECIAL_PURPOSE_H_
