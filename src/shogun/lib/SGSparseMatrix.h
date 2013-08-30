/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Written (W) 2010,2012 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 * Copyright (C) 2012 Soeren Sonnenburg
 */

#ifndef __SGSPARSEMATRIX_H__
#define __SGSPARSEMATRIX_H__

#include <shogun/lib/common.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGReferencedData.h>

namespace shogun
{

template <class T> class SGSparseVector;
class CFile;
class CRegressionLabels;

/** @brief template class SGSparseMatrix */
template <class T> class SGSparseMatrix : public SGReferencedData
{
	public:
		/** default constructor */
		SGSparseMatrix();

		/** constructor for setting params */
		SGSparseMatrix(SGSparseVector<T>* vecs, index_t num_feat,
				index_t num_vec, bool ref_counting=true);

		/** constructor to create new matrix in memory */
		SGSparseMatrix(index_t num_feat, index_t num_vec, bool ref_counting=true);

		/** copy constructor */
		SGSparseMatrix(const SGSparseMatrix &orig);

		/** destructor */
		virtual ~SGSparseMatrix();

		/** index access operator */
		inline const SGSparseVector<T>& operator[](index_t index) const
		{
			return sparse_matrix[index];
		}

		/** index access operator */
		inline SGSparseVector<T>& operator[](index_t index)
		{
			return sparse_matrix[index];
		}

		/** 
		 * get the sparse matrix (no copying is done here)
		 *
		 * @return the refcount increased matrix
		 */
		inline SGSparseMatrix<T> get()
		{
			return *this;
		}

		/** compute sparse-matrix dense-vector multiplication
		 * @param v the dense-vector to be multiplied with
		 * @return the result vector \f$Q*v\f$, Q being this sparse matrix
		 */
		const SGVector<T> operator*(SGVector<T> v) const
		{
			SGVector<T> result(num_vectors);
			REQUIRE(v.vlen==num_features,
				"Dimension mismatch! %d vs %d\n",
				v.vlen, num_features);
			for (index_t i=0; i<num_vectors; ++i)
				result[i]=sparse_matrix[i].dense_dot(1.0, v.vector, v.vlen, 0.0);

			return result;
		}

		/** compute sparse-matrix dense-vector multiplication
		 * @param v the dense-vector to be multiplied with
		 * @return the result vector \f$Q*v\f$, Q being this sparse matrix
		 */
		template<class ST> const SGVector<T> operator*(SGVector<ST> v) const;

		/** load sparse matrix from file
		 *
		 * @param loader File object via which to load data
		 */
		void load(CFile* loader);

		SGSparseMatrix<T> get_transposed();
		CRegressionLabels* load_svmlight_file(char* fname, bool do_sort_features);
		bool write_svmlight_file(char* fname, CRegressionLabels* label);
		void sort_features();

		/** save sparse matrix to file
		 *
		 * @param saver File object via which to save data
		 */
		void save(CFile* saver);

protected:

		/** copy data */
		virtual void copy_data(const SGReferencedData& orig);

		/** init data */
		virtual void init_data();

		/** free data */
		virtual void free_data();

public:

	/// total number of vectors
	index_t num_vectors;

	/// total number of features
	index_t num_features;

	/// array of sparse vectors of size num_vectors
	SGSparseVector<T>* sparse_matrix;

};
}
#endif // __SGSPARSEMATRIX_H__
