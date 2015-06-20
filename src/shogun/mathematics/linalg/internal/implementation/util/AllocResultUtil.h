/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Soumyajit De
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

#ifndef ALLOC_RESULT_UTIL_H_
#define ALLOC_RESULT_UTIL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUMatrix.h>
#endif // HAVE_VIENNACL

namespace shogun
{

namespace linalg
{

namespace util
{

/**
 * @brief Template struct allocate_result for allocating objects of return type
 * for element-wise operations. This generic version takes care of the vector
 * types supported by Shogun (SGVector and CGPUVector).
 */
template <class Operand, class ReturnType>
struct allocate_result
{
	/**
	 * Creates a newly allocated memory for ReturnType, assuming that the
	 * ReturnType is a vector.
	 *
	 * @param op The operand for the element-wise operation
	 * @return The newly allocated result of ReturnType
	 */
	static ReturnType alloc(Operand op)
	{
		return ReturnType(op.size());
	}
};

/**
 * @brief Specialization for allocate_result when return type is SGMatrix. Works
 * with different scalar types as well. T defines the scalar type for the operand
 * and whereas ST is the scalar type for the result of the element-wise operation.
 */
template <typename T, typename ST>
struct allocate_result<SGMatrix<T>,SGMatrix<ST>>
{
	/**
	 * Creates a newly allocated memory for SGMatrix.
	 *
	 * @param m The operand for the element-wise operation of scalar type T
	 * @return The newly allocated result of SGMatrix of scalar type ST
	 */
	static SGMatrix<ST> alloc(SGMatrix<T> m)
	{
		return SGMatrix<ST>(m.num_rows, m.num_cols);
	}
};

#ifdef HAVE_VIENNACL
/**
 * @brief Specialization for allocate_result when return type is CGPUMatrix. Works
 * with different scalar types as well. T defines the scalar type for the operand
 * and whereas ST is the scalar type for the result of the element-wise operation.
 */
template <typename T, typename ST>
struct allocate_result<CGPUMatrix<T>, CGPUMatrix<ST>>
{
	/**
	 * Creates a newly allocated memory for CGPUMatrix.
	 *
	 * @param m The operand for the element-wise operation of scalar type T
	 * @return The newly allocated result of CGPUMatrix of scalar type ST
	 */
	static CGPUMatrix<ST> alloc(CGPUMatrix<T> m)
	{
		return CGPUMatrix<ST>(m.num_rows, m.num_cols);
	}
};
#endif // HAVE_VIENNACL

}

}

}
#endif // ALLOC_RESULT_UTIL_H_
