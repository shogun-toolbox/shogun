/*
 * Copyright (c) 2017, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
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
 */

#ifndef LINALG_ENUMS_H__
#define LINALG_ENUMS_H__

#include <type_traits>
#include <shogun/lib/common.h>

namespace shogun
{

	namespace linalg
	{
		template <typename T, typename U>
		struct promote
		{
			constexpr static bool is_one_complex =
				std::is_same<T, complex128_t>::value ||
				std::is_same<U, complex128_t>::value;

 			constexpr static bool is_one_floating =
				std::is_floating_point<T>::value ||
				std::is_floating_point<U>::value;

 			using complex_type =
				std::conditional_t<std::is_same<T, complex128_t>::value, T, U>;
			using floating_type =
				std::conditional_t<std::is_floating_point<T>::value, T, U>;
			using bigger_type = std::conditional_t<(sizeof(T) > sizeof(U)), T, U>;

 			using type = std::conditional_t < is_one_complex, complex_type,
				std::conditional_t<is_one_floating, floating_type, bigger_type>>;
		};
		
		/**
		 * Enum for choosing the algorithm used to calculate SVD.
		 * The <em>bidiagonal divide and conquer</em> algorithm
		 * is faster on large matrices but it's available with
		 * Eigen >= 3.3, furthermore it may produce inaccurate
		 * results when compiled with unsafe math optimization.
		 * For more details see:
		 * https://eigen.tuxfamily.org/dox/classEigen_1_1BDCSVD.html
		 * https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
		 */
		enum class SVDAlgorithm
		{
			BidiagonalDivideConquer,
			Jacobi
		};
	}
}

#endif // LINALG_ENUMS_H__