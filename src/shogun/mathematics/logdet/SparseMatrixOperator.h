/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#ifndef SPARSE_MATRIX_OPERATOR_H_
#define SPARSE_MATRIX_OPERATOR_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/mathematics/logdet/MatrixOperator.h>

namespace shogun
{
template<class T> class SGVector;
template<class T> class SGSparseMatrix;

/** @brief Class that represents a sparse-matrix linear operator.
 * It computes matrix-vector product \f$Ax\f$ in its apply method,
 * \f$A\in\mathbb{C}^{m\times n},A:\mathbb{C}^{n}\rightarrow \mathbb{C}^{m}\f$
 * being the matrix operator and \f$x\in\mathbb{C}^{n}\f$ being the vector.
 * The result is a vector \f$y\in\mathbb{C}^{m}\f$.
 */
template<class T> class CSparseMatrixOperator : public CMatrixOperator<T>
{
/** this class has support for complex64_t */
typedef bool supports_complex64_t;

public:
	/** default constructor */
	CSparseMatrixOperator();

	/** 
	 * constructor
	 *
	 * @param op the sparse matrix to be used as the linear operator
	 */
	explicit CSparseMatrixOperator(SGSparseMatrix<T> op);

	/**
	 * copy constructor that creates a deep copy
	 *
	 * @param orig the original sparse matrix operator
	 */
	CSparseMatrixOperator(const CSparseMatrixOperator<T>& orig);

	/** destructor */
	~CSparseMatrixOperator();

	/** 
	 * method that applies the sparse-matrix linear operator to a vector
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

	/** @return the sparse matrix operator */
	SGSparseMatrix<T> get_matrix_operator() const;

	/**
	 * create a new sparse matrix operator of Scalar type
	 */
	template<class Scalar>
	inline operator CSparseMatrixOperator<Scalar>*() const
	{
		REQUIRE(m_operator.sparse_matrix, "Matrix is not initialized!\n");
		typedef SGSparseVector<Scalar> vector;
		typedef SGSparseVectorEntry<Scalar> entry;

		SGSparseMatrix<Scalar> casted_m(m_operator.num_vectors, m_operator.num_features);

		vector* rows=SG_MALLOC(vector, m_operator.num_features);
		for (index_t i=0; i<m_operator.num_vectors; ++i)
		{
			entry* features=SG_MALLOC(entry, m_operator[i].num_feat_entries);
			for (index_t j=0; j<m_operator[i].num_feat_entries; ++j)
			{
				features[j].feat_index=m_operator[i].features[j].feat_index;
				features[j].entry=static_cast<Scalar>(m_operator[i].features[j].entry);
			}
			rows[i].features=features;
			rows[i].num_feat_entries=m_operator[i].num_feat_entries;
		}
		casted_m.sparse_matrix=rows;

		SG_SDEBUG("SparseMatrixOperator::static_cast(): Creating casted operator!\n");

		return new CSparseMatrixOperator<Scalar>(casted_m);
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "SparseMatrixOperator";
	}

private:
	/** the sparse matrix operator */
	SGSparseMatrix<T> m_operator;

	/** initialize with default values and register params */
	void init();

};

}
#endif // SPARSE_MATRIX_OPERATOR_H_
