/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#ifndef LINEAR_OPERATOR_H_
#define LINEAR_OPERATOR_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>

namespace shogun
{
template<class T> class SGVector;

/** @brief Abstract template base class that represents a linear operator,
 *  e.g. a matrix
 */
template<class T> class CLinearOperator : public CSGObject
{
public:
	/** default constructor */
	CLinearOperator()
	: CSGObject()
	{
		init();

		SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
	}

	/** 
	 * constructor
	 *
	 * @param dimension dimension of the vector on which the operator can be applied
	 */
	CLinearOperator(index_t dimension)
	: CSGObject()
	{
		init();
	
		m_dimension=dimension;

		SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
	}

	/** destructor */
	virtual ~CLinearOperator()
	{
		SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
	}

	/** @return the dimension on which the linear operator can apply */
	const index_t get_dimension() const
	{
		return m_dimension;
	}

	/** 
	 * abstract method that applies the linear operator to a vector
	 *
	 * @param b the vector to which the linear operator applies
	 * @return the result vector
	 */
	virtual SGVector<T> apply(SGVector<T> b) const = 0;

	/** @return object name */
	virtual const char* get_name() const
	{
		return "LinearOperator";
	}
protected:
	/** the dimension of vector on which the linear operator can apply */
	index_t m_dimension;

private:
	/** initialize with default values and register params */
	void init()
	{
		m_dimension=0;

		SG_ADD(&m_dimension, "dimension",
			"Dimension of the vector on which linear operator can apply",
			MS_NOT_AVAILABLE);
	}

};

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
template class CLinearOperator<complex64_t>;
}

#endif // LINEAR_OPERATOR_H_
