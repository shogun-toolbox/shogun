/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/lib/computation/jobresult/VectorResult.h>

namespace shogun
{
template class CVectorResult<bool>;
template class CVectorResult<char>;
template class CVectorResult<int8_t>;
template class CVectorResult<uint8_t>;
template class CVectorResult<int16_t>;
template class CVectorResult<uint16_t>;
template class CVectorResult<int32_t>;
template class CVectorResult<uint32_t>;
template class CVectorResult<int64_t>;
template class CVectorResult<uint64_t>;
template class CVectorResult<float32_t>;
template class CVectorResult<float64_t>;
template class CVectorResult<floatmax_t>;
template class CVectorResult<complex128_t>;
}
