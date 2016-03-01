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

#include <shogun/mathematics/linalg/linop/MatrixOperator.h>

namespace shogun
{
template<class T> class SGVector;
template<class T> class SGMatrix;

/** @brief Class that represents a dense-matrix linear operator.
 * It computes matrix-vector product \f$Ax\f$ in its apply method,
 * \f$A\in\mathbb{C}^{m\times n},A:\mathbb{C}^{n}\rightarrow \mathbb{C}^{m}\f$
 * being the matrix operator and \f$x\in\mathbb{C}^{n}\f$ being the vector.
 * The result is a vector \f$y\in\mathbb{C}^{m}\f$.
 */
template<class T> class CDenseMatrixOperator : public CMatrixOperator<T>
{
/** this class has support for complex128_t */
typedef bool supports_complex128_t;

public:
	/** default constructor */
	CDenseMatrixOperator();

	/**
	 * constructor
	 *
	 * @param op the dense matrix to be used as the linear operator
	 */
	explicit CDenseMatrixOperator(SGMatrix<T> op);

	/**
	 * copy constructor that creates a deep copy
	 *
	 * @param orig the original dense matrix operator
	 */
	CDenseMatrixOperator(const CDenseMatrixOperator<T>& orig);

	/** destructor */
	~CDenseMatrixOperator();

	/**
	 * method that applies the dense-matrix linear operator to a vector
	 *
	 * @param b the vector to which the linear operator applies
	 * @return the result vector
	 */
	virtual SGVector<T> apply(SGVector<T> b) const;

	/**
	 * method that sets the main diagonal of the matrix
	 *
	 * @param diag the diagonal to be set
	 */
	virtual void set_diagonal(SGVector<T> diag);

	/**
	 * method that returns the main diagonal of the matrix
	 *
	 * @return the diagonal
	 */
	virtual SGVector<T> get_diagonal() const;

	/** @return the dense matrix operator */
	SGMatrix<T> get_matrix_operator() const;

	/**
	 * create a new dense matrix operator of Scalar type
	 */
	template<class Scalar>
	inline operator CDenseMatrixOperator<Scalar>*() const
	{
		REQUIRE(m_operator.matrix, "Matrix is not initialized!\n");

		SGMatrix<Scalar> casted_m(m_operator.num_rows, m_operator.num_cols);
		for (index_t i=0; i<m_operator.num_cols; ++i)
		{
			for (index_t j=0; j<m_operator.num_rows; ++j)
				casted_m(j,i)=static_cast<Scalar>(m_operator(j,i));
		}

		return new CDenseMatrixOperator<Scalar>(casted_m);
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "DenseMatrixOperator";
	}

private:
	/** the dense matrix operator */
	SGMatrix<T> m_operator;

	/** initialize with default values and register params */
	void init();

};

}

#endif // DENSE_MATRIX_OPERATOR_H_
