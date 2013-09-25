/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 * Written (W) 2013 Heiko Strathmann
 *
 */

#include <shogun/lib/computation/jobresult/ScalarResult.h>

namespace shogun
{
template class CScalarResult<bool>;
template class CScalarResult<char>;
template class CScalarResult<int8_t>;
template class CScalarResult<uint8_t>;
template class CScalarResult<int16_t>;
template class CScalarResult<uint16_t>;
template class CScalarResult<int32_t>;
template class CScalarResult<uint32_t>;
template class CScalarResult<int64_t>;
template class CScalarResult<uint64_t>;
template class CScalarResult<float32_t>;
template class CScalarResult<float64_t>;
template class CScalarResult<floatmax_t>;
template class CScalarResult<complex128_t>;
}
