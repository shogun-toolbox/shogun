/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Björn Esser
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
