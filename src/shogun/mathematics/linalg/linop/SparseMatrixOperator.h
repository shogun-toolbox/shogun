/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Bjoern Esser
 */

#ifndef SPARSE_MATRIX_OPERATOR_H_
#define SPARSE_MATRIX_OPERATOR_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/mathematics/linalg/linop/MatrixOperator.h>

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
			io::print("Row number {}. Number of Non-zeros {}. Colums ", i, nnzs);
			for(index_t j=1; j<=nnzs; ++j)
			{
				io::print("{}", m_ptr[i][j]);
				if (j<nnzs)
					io::print(", ");
			}
			io::print("\n");
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
template<class T> class SparseMatrixOperator : public MatrixOperator<T>
{
/** this class has support for complex128_t */
typedef bool supports_complex128_t;

public:
	/** default constructor */
	SparseMatrixOperator();

	/**
	 * constructor
	 *
	 * @param op the sparse matrix to be used as the linear operator
	 */
	explicit SparseMatrixOperator(SGSparseMatrix<T> op);

	/**
	 * copy constructor that creates a deep copy
	 *
	 * @param orig the original sparse matrix operator
	 */
	SparseMatrixOperator(const SparseMatrixOperator<T>& orig);

	/** destructor */
	~SparseMatrixOperator() override;

	/**
	 * method that applies the sparse-matrix linear operator to a vector
	 *
	 * @param b the vector to which the linear operator applies
	 * @return the result vector
	 */
	SGVector<T> apply(SGVector<T> b) const override;

	/**
	 * method that sets the main diagonal of the matrix
	 *
	 * @param diag the diagonal to be set
	 */
	void set_diagonal(SGVector<T> diag) override;

	/**
	 * method that returns the main diagonal of the matrix
	 *
	 * @return the diagonal
	 */
	SGVector<T> get_diagonal() const override;

	/** @return the sparse matrix operator */
	SGSparseMatrix<T> get_matrix_operator() const;

	/** @return the sparsity structure of matrix power */
	SparsityStructure* get_sparsity_structure(int64_t power=1) const;

	/**
	 * create a new sparse matrix operator of Scalar type
	 */
	template<class Scalar>
	inline operator SparseMatrixOperator<Scalar>*() const
	{
		require(m_operator.sparse_matrix, "Matrix is not initialized!");
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

		return new SparseMatrixOperator<Scalar>(casted_m);
	}

	/** @return object name */
	const char* get_name() const override
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
