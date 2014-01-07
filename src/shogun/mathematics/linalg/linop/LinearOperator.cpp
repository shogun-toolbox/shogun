/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 * Written (W) 2013 Heiko Strathmann
 */

#include <mathematics/linalg/linop/LinearOperator.h>

namespace shogun
{
template class CLinearOperator<bool>;
template class CLinearOperator<char>;
template class CLinearOperator<int8_t>;
template class CLinearOperator<uint8_t>;
template class CLinearOperator<int16_t>;
template class CLinearOperator<uint16_t>;
template class CLinearOperator<int32_t>;
template class CLinearOperator<uint32_t>;
template class CLinearOperator<int64_t>;
template class CLinearOperator<uint64_t>;
template class CLinearOperator<float32_t>;
template class CLinearOperator<float64_t>;
template class CLinearOperator<floatmax_t>;
template class CLinearOperator<complex128_t>;
}
