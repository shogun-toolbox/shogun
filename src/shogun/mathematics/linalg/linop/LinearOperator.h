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

namespace shogun
{

/** @brief Abstract template base class that represents a linear operator,
 *  e.g. a matrix
 */
template<class T>
class CLinearOperator : public CSGObject
{
public:
	/** default constructor */
	CLinearOperator();

	/**
	 * constructor
	 *
	 * @param dimension dimension of the vector on which the operator can be applied
	 */
	CLinearOperator(index_t dimension);

	/** destructor */
	virtual ~CLinearOperator();

	/** @return the dimension on which the linear operator can apply */
	const index_t get_dimension() const;

	/**
	 * abstract method that applies the linear operator to Operand(eg. a vector)
	 *
	 * @param b the Operand to which the linear operator applies
	 * @return the result(eg. a vector)
	 */
	virtual SGVector<T> apply(SGVector<T> b) const=0;

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
	void init();

};

}

#endif // LINEAR_OPERATOR_H_
