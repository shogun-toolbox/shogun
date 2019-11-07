/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Sunil Mahendrakar, Bjoern Esser
 */

#include <shogun/mathematics/linalg/linop/LinearOperator.h>

namespace shogun
{

template<class T>
LinearOperator<T>::LinearOperator() : SGObject()
{
	init();
}

template<class T>
LinearOperator<T>::LinearOperator(index_t dimension) : SGObject()
{
	init();

	m_dimension=dimension;
}

template<class T>
LinearOperator<T>::~LinearOperator()
{
}

template<class T>
const index_t LinearOperator<T>::get_dimension() const
{
	return m_dimension;
}

template<class T>
void LinearOperator<T>::init()
{
	m_dimension=0;

	SG_ADD(&m_dimension, "dimension",
		"Dimension of the vector on which linear operator can apply");
}

template class LinearOperator<bool>;
template class LinearOperator<char>;
template class LinearOperator<int8_t>;
template class LinearOperator<uint8_t>;
template class LinearOperator<int16_t>;
template class LinearOperator<uint16_t>;
template class LinearOperator<int32_t>;
template class LinearOperator<uint32_t>;
template class LinearOperator<int64_t>;
template class LinearOperator<uint64_t>;
template class LinearOperator<float32_t>;
template class LinearOperator<float64_t>;
template class LinearOperator<floatmax_t>;
template class LinearOperator<complex128_t>;

}
