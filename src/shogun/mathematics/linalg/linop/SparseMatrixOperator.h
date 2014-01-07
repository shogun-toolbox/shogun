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

#include <lib/config.h>
#include <lib/SGSparseVector.h>
#include <lib/SGSparseMatrix.h>
#include <mathematics/linalg/linop/MatrixOperator.h>

namespace shogun
{
template<class T> class SGVector;
template<class T> class SGSparseMatrix;

/** @brief Struct that represents the sparsity structure of the Sparse Matrix
 * in CRS. Implementation has been adapted from Krylstat (https://github.com/
 * Froskekongen/KRYLSTAT) library (c) Erlend Aune <erlenda@math.ntnu.no>
 * under GPL2+
 */
struct SparsityStructure
{
	/** default constructor */
	SparsityStructure() : m_num_rows(0), m_ptr(NULL) {}

	/**
	 * constructor
	 *
	 * @param row_offsets outer index ptr in CRS
	 * @param column_indices inner index ptr in CRS
	 * @param num_rows number of rows
	 */
	SparsityStructure(index_t* row_offsets, index_t* column_indices,
		index_t num_rows)
	: m_num_rows(num_rows),
		m_ptr(new int32_t*[num_rows]())
	{
		for (index_t i=0; i<m_num_rows; ++i)
		{
			index_t current_index=row_offsets[i];
			index_t new_index=row_offsets[i+1];
			index_t length_row=(new_index-current_index);

			m_ptr[i]=new int32_t[length_row+1]();
			m_ptr[i][0]=length_row;

			for (index_t j=1; j<=length_row; ++j)
				m_ptr[i][j]=column_indices[current_index++];
		}
	}

	/** destructor */
	~SparsityStructure()
	{
		for (index_t i=0; i<m_num_rows; ++i)
			delete [] m_ptr[i];
		delete [] m_ptr;
	}

	/** display sparsity structure */
	void display_sparsity_structure()
	{
		for (index_t i=0; i<m_num_rows; ++i)
		{
			index_t nnzs=m_ptr[i][0];
			SG_SPRINT("Row number %d. Number of Non-zeros %d. Colums ", i, nnzs);
			for(index_t j=1; j<=nnzs; ++j)
			{
				SG_SPRINT("%d", m_ptr[i][j]);
				if (j<nnzs)
					SG_SPRINT(", ");
			}
			SG_SPRINT("\n");
		}
	}

	/** number of rows */
	index_t m_num_rows;

	/** the pointer that stores the nnz entries */
	int32_t **m_ptr;
};


/** @brief Class that represents a sparse-matrix linear operator.
 * It computes matrix-vector product \f$Ax\f$ in its apply method,
 * \f$A\in\mathbb{C}^{m\times n},A:\mathbb{C}^{n}\rightarrow \mathbb{C}^{m}\f$
 * being the matrix operator and \f$x\in\mathbb{C}^{n}\f$ being the vector.
 * The result is a vector \f$y\in\mathbb{C}^{m}\f$.
 */
template<class T> class CSparseMatrixOperator : public CMatrixOperator<T>
{
/** this class has support for complex128_t */
typedef bool supports_complex128_t;

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

	/** @return the sparsity structure of matrix power */
	SparsityStructure* get_sparsity_structure(int64_t power=1) const;

	/**
	 * create a new sparse matrix operator of Scalar type
	 */
	template<class Scalar>
	inline operator CSparseMatrixOperator<Scalar>*() const
	{
		REQUIRE(m_operator.sparse_matrix, "Matrix is not initialized!\n");
		typedef SGSparseVector<Scalar> vector;
		typedef SGSparseVectorEntry<Scalar> entry;

		vector* rows=SG_MALLOC(vector, m_operator.num_vectors);

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

		SGSparseMatrix<Scalar> casted_m;
		casted_m.sparse_matrix=rows;
		casted_m.num_vectors=m_operator.num_vectors;
		casted_m.num_features= m_operator.num_features;

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
