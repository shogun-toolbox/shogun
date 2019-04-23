/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Bjoern Esser
 */

#include <shogun/mathematics/linalg/linop/MatrixOperator.h>

namespace shogun
{
template class MatrixOperator<bool>;
template class MatrixOperator<char>;
template class MatrixOperator<int8_t>;
template class MatrixOperator<uint8_t>;
template class MatrixOperator<int16_t>;
template class MatrixOperator<uint16_t>;
template class MatrixOperator<int32_t>;
template class MatrixOperator<uint32_t>;
template class MatrixOperator<int64_t>;
template class MatrixOperator<uint64_t>;
template class MatrixOperator<float32_t>;
template class MatrixOperator<float64_t>;
template class MatrixOperator<floatmax_t>;
template class MatrixOperator<complex128_t>;
}
