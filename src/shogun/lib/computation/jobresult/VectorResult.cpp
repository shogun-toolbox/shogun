/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Björn Esser
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
