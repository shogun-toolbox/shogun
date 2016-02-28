/*
 * Copyright (c) 2016, Shogun Toolbox Foundation
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
 * Written (W) 2016 Heiko Strathmann
 */
#include <shogun/base/Parameter.h>
#include <shogun/lib/SGVectorObject.h>
#include <shogun/lib/SGVector.h>

using namespace shogun;

template<class T> CSGVectorObject<T>::CSGVectorObject() : CSGObject()
{
	init();
}

template<class T> CSGVectorObject<T>::CSGVectorObject(SGVector<T> vec) : CSGObject()
{
	m_vec = vec;

	init();
}


template<class T> void CSGVectorObject<T>::init()
{
	SG_ADD(&m_vec, "Vector", "Wrapped SGVector", MS_NOT_AVAILABLE);
}

template class CSGVectorObject<bool>;
template class CSGVectorObject<char>;
template class CSGVectorObject<int8_t>;
template class CSGVectorObject<uint8_t>;
template class CSGVectorObject<int16_t>;
template class CSGVectorObject<uint16_t>;
template class CSGVectorObject<int32_t>;
template class CSGVectorObject<uint32_t>;
template class CSGVectorObject<int64_t>;
template class CSGVectorObject<uint64_t>;
template class CSGVectorObject<float32_t>;
template class CSGVectorObject<float64_t>;
template class CSGVectorObject<floatmax_t>;
