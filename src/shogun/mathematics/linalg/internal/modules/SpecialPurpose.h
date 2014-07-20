/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
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

/** Contains special purpose, algorithm specific functions. Uses the same 
 * backend as the Core module 
 */
#ifndef SPECIAL_PURPOSE_H_
#define SPECIAL_PURPOSE_H_

#include <shogun/mathematics/linalg/internal/implementation/SpecialPurpose.h>

namespace shogun
{

namespace linalg
{

namespace special_purpose
{
	
/** Applies the elementwise logistic function f(x) = 1/(1+exp(-x)) to a matrix */
template <Backend backend=linalg_traits<Core>::backend,class Matrix>
void logistic(Matrix A, Matrix result)
{
	implementation::special_purpose::logistic<backend, Matrix>::compute(A, result);
}

/** Performs the operation C(i,j) = C(i,j) * A(i,j) * (1.0-A(i,j) for all i and j*/ 
template <Backend backend=linalg_traits<Core>::backend,class Matrix>
void multiply_by_logistic_derivative(Matrix A, Matrix C)
{
	implementation::special_purpose::multiply_by_logistic_derivative<backend, Matrix>::compute(A, C);
}

}

}

}
#endif // SPECIAL_PURPOSE_H_
