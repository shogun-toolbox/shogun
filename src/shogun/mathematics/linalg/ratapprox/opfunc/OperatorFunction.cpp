/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Bjoern Esser
 */

#include <shogun/mathematics/linalg/ratapprox/opfunc/OperatorFunction.h>

namespace shogun
{
template class COperatorFunction<bool>;
template class COperatorFunction<char>;
template class COperatorFunction<int8_t>;
template class COperatorFunction<uint8_t>;
template class COperatorFunction<int16_t>;
template class COperatorFunction<uint16_t>;
template class COperatorFunction<int32_t>;
template class COperatorFunction<uint32_t>;
template class COperatorFunction<int64_t>;
template class COperatorFunction<uint64_t>;
template class COperatorFunction<float32_t>;
template class COperatorFunction<float64_t>;
template class COperatorFunction<floatmax_t>;
template class COperatorFunction<complex128_t>;
}
