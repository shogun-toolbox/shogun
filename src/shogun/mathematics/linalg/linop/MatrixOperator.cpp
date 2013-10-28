/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/mathematics/linalg/linop/MatrixOperator.h>

namespace shogun
{
template class CMatrixOperator<bool>;
template class CMatrixOperator<char>;
template class CMatrixOperator<int8_t>;
template class CMatrixOperator<uint8_t>;
template class CMatrixOperator<int16_t>;
template class CMatrixOperator<uint16_t>;
template class CMatrixOperator<int32_t>;
template class CMatrixOperator<uint32_t>;
template class CMatrixOperator<int64_t>;
template class CMatrixOperator<uint64_t>;
template class CMatrixOperator<float32_t>;
template class CMatrixOperator<float64_t>;
template class CMatrixOperator<floatmax_t>;
template class CMatrixOperator<complex128_t>;
}
