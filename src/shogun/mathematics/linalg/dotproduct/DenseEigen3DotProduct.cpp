/*
 * Copyright (c) 2014, Shogun Toolbox Foundation
 * All rights reserved.
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
 * Written (W) 2014 Sunil K. Mahendrakar
 */

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include<shogun/mathematics/linalg/dotproduct/VectorDotProduct.h>
#include<shogun/mathematics/linalg/dotproduct/DenseEigen3DotProduct.h>
namespace shogun
{
template<class T>
T CDenseEigen3DotProduct<T>::compute(SGVector<T> vector1, SGVector<T> vector2) const
{
    //if (vector1.vlen != vector2.vlen)
    //SG_ERROR("dimension mismatch")
    Map<Matrix<T, Dynamic, 1> > vec1(vector1.vector, vector1.vlen);
    Map<Matrix<T, Dynamic, 1> > vec2(vector2.vector, vector2.vlen);
    return vec1.dot(vec2);
}

template class CDenseEigen3DotProduct<bool>;
template class CDenseEigen3DotProduct<char>;
template class CDenseEigen3DotProduct<int8_t>;
template class CDenseEigen3DotProduct<uint8_t>;
template class CDenseEigen3DotProduct<int16_t>;
template class CDenseEigen3DotProduct<uint16_t>;
template class CDenseEigen3DotProduct<int32_t>;
template class CDenseEigen3DotProduct<uint32_t>;
template class CDenseEigen3DotProduct<int64_t>;
template class CDenseEigen3DotProduct<uint64_t>;
template class CDenseEigen3DotProduct<float32_t>;
template class CDenseEigen3DotProduct<float64_t>;
template class CDenseEigen3DotProduct<floatmax_t>;
template class CDenseEigen3DotProduct<complex128_t>;
}
#endif //HAVE_EIGEN3
