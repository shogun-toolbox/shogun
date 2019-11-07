/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Bjoern Esser
 */

#include <shogun/mathematics/linalg/ratapprox/opfunc/OperatorFunction.h>

namespace shogun
{
template class OperatorFunction<bool>;
template class OperatorFunction<char>;
template class OperatorFunction<int8_t>;
template class OperatorFunction<uint8_t>;
template class OperatorFunction<int16_t>;
template class OperatorFunction<uint16_t>;
template class OperatorFunction<int32_t>;
template class OperatorFunction<uint32_t>;
template class OperatorFunction<int64_t>;
template class OperatorFunction<uint64_t>;
template class OperatorFunction<float32_t>;
template class OperatorFunction<float64_t>;
template class OperatorFunction<floatmax_t>;
template class OperatorFunction<complex128_t>;
}
