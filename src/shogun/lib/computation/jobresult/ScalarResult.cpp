/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Björn Esser
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
