/*
 * Copyright (c) 2014, Shogun Toolbox Foundation
 * All rights reserved.
 *
 * Written (W) 2014 Sunil K. Mahendrakar
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
 */

#include <shogun/lib/config.h>
#include <shogun/mathematics/linalg/dotproduct/VectorDotOperator.h>
#include<shogun/lib/SGVector.h>

namespace shogun
{
template class VectorDotOperator<int8_t, SGVector<int8_t> >;
template class VectorDotOperator<uint8_t, SGVector<uint8_t> >;
template class VectorDotOperator<int16_t, SGVector<int16_t> >;
template class VectorDotOperator<uint16_t, SGVector<uint16_t> >;
template class VectorDotOperator<int32_t, SGVector<int32_t> >;
template class VectorDotOperator<uint32_t, SGVector<uint32_t> >;
template class VectorDotOperator<int64_t, SGVector<int64_t> >;
template class VectorDotOperator<uint64_t, SGVector<uint64_t> >;
template class VectorDotOperator<float32_t, SGVector<float32_t> >;
template class VectorDotOperator<float64_t, SGVector<float64_t> >;
template class VectorDotOperator<floatmax_t, SGVector<floatmax_t> >;
template class VectorDotOperator<complex128_t, SGVector<complex128_t> >;

}
