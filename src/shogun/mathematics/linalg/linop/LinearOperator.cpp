/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Sunil Mahendrakar, Bjoern Esser
 */

#include <shogun/mathematics/linalg/linop/LinearOperator.h>
#include <shogun/base/Parameter.h>

namespace shogun
{

template<class T>
CLinearOperator<T>::CLinearOperator() : CSGObject()
{
	init();
}

template<class T>
CLinearOperator<T>::CLinearOperator(index_t dimension) : CSGObject()
{
	init();

	m_dimension=dimension;
}

template<class T>
CLinearOperator<T>::~CLinearOperator()
{
}

template<class T>
const index_t CLinearOperator<T>::get_dimension() const
{
	return m_dimension;
}

template<class T>
void CLinearOperator<T>::init()
{
	m_dimension=0;

	SG_ADD(&m_dimension, "dimension",
		"Dimension of the vector on which linear operator can apply");
}

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
