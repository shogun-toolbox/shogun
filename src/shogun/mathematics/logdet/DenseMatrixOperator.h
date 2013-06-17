/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#ifndef DENSE_MATRIX_OPERATOR_H_
#define DENSE_MATRIX_OPERATOR_H_

#include <shogun/lib/config.h>
#include <shogun/mathematics/logdet/LinearOperator.h>

namespace shogun
{
template<class T> class SGVector;
template<class T> class SGMatrix;

/** @brief Class that represents a dense-matrix linear operator */
template<class T> class CDenseMatrixOperator : public CLinearOperator<T>
{
public:
	/** default constructor, no args */
	CDenseMatrixOperator();

	/** default constructor, one args */
	CDenseMatrixOperator(SGMatrix<T> op);

	/** destructor */
	~CDenseMatrixOperator();

	/** method that applies the dense-matrix linear operator to a vector
	 * @param b the vector to which the linear operator applies
	 * @return the result vector
	 */
	virtual SGVector<T> apply(SGVector<T> b) const;

	/** @return the dense matrix operator */
	SGMatrix<T> get_matrix_operator() const;

	/** @return object name */
	virtual const char* get_name() const
	{
		return "CDenseMatrixOperator";
	}

private:
	/** the dense matrix operator */
	const SGMatrix<T> m_operator;
};

}

#endif // LINEAR_OPERATOR_H_
