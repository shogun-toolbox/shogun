/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/mathematics/linalg/linop/LinearOperator.h>

namespace shogun
{
template class CLinearOperator<SGVector<bool>, SGVector<bool> >;
template class CLinearOperator<SGVector<char>, SGVector<char> >;
template class CLinearOperator<SGVector<int8_t>, SGVector<int8_t> >;
template class CLinearOperator<SGVector<uint8_t>, SGVector<uint8_t> >;
template class CLinearOperator<SGVector<int16_t>, SGVector<int16_t> >;
template class CLinearOperator<SGVector<uint16_t>, SGVector<uint16_t> >;
template class CLinearOperator<SGVector<int32_t>, SGVector<int32_t> >;
template class CLinearOperator<SGVector<uint32_t>, SGVector<uint32_t> >;
template class CLinearOperator<SGVector<int64_t>, SGVector<int64_t> >;
template class CLinearOperator<SGVector<uint64_t>, SGVector<uint64_t> >;
template class CLinearOperator<SGVector<float32_t>, SGVector<float32_t> >;
template class CLinearOperator<SGVector<float64_t>, SGVector<float64_t> >;
template class CLinearOperator<SGVector<floatmax_t>, SGVector<floatmax_t> >;
template class CLinearOperator<SGVector<complex128_t>, SGVector<complex128_t> >;
}
