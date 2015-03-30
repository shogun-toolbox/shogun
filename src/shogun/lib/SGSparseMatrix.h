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

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/lib/SGReferencedData.h>
#include <shogun/lib/SGVector.h>
#include <shogun/io/SGIO.h>

namespace shogun
{

template <class T> class SGSparseVector;
template <class ST> struct SGSparseVectorEntry;
template<class T> class SGMatrix;
class CFile;
class CLibSVMFile;

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

		/** constructor to create new sparse matrix from a dense one
		 *
		 * @param dense dense matrix to be converted
		 */
		SGSparseMatrix(SGMatrix<T> dense);

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

		/** operator overload for sparse-matrix read only access
		 * @param i_row
		 * @param i_col
		 */
		inline const T operator()(index_t i_row, index_t i_col) const
		{
			REQUIRE(i_row>=0, "Provided row index %d negative!\n", i_row);
			REQUIRE(i_col>=0, "Provided column index %d negative!\n", i_col);
			REQUIRE(i_row<num_features, "Provided row index (%d) is larger than number of rows (%d)\n",
							i_row, num_features);
			REQUIRE(i_col<num_vectors, "Provided column index (%d) is larger than number of columns (%d)\n",
							i_col, num_vectors);

			for (index_t i=0; i<sparse_matrix[i_col].num_feat_entries; ++i)
			{
				if (i_row==sparse_matrix[i_col].features[i].feat_index)
					return sparse_matrix[i_col].features[i].entry;
			}
			return 0;
		}

		/** operator overload for sparse-matrix r/w access
		 * @param i_row
		 * @param i_col
		 */
		inline T& operator()(index_t i_row, index_t i_col)
		{
			REQUIRE(i_row>=0, "Provided row index %d negative!\n", i_row);
			REQUIRE(i_col>=0, "Provided column index %d negative!\n", i_col);
			REQUIRE(i_row<num_features, "Provided row index (%d) is larger than number of rows (%d)\n",
							i_row, num_features);
			REQUIRE(i_col<num_vectors, "Provided column index (%d) is larger than number of columns (%d)\n",
							i_col, num_vectors);

			for (index_t i=0; i<sparse_matrix[i_col].num_feat_entries; ++i)
			{
				if (i_row==sparse_matrix[i_col].features[i].feat_index)
					return sparse_matrix[i_col].features[i].entry;
			}
			index_t j=sparse_matrix[i_col].num_feat_entries;
			sparse_matrix[i_col].num_feat_entries=j+1;
			sparse_matrix[i_col].features=SG_REALLOC(SGSparseVectorEntry<T>,
				sparse_matrix[i_col].features, j, j+1);
			sparse_matrix[i_col].features[j].feat_index=i_row;
			sparse_matrix[i_col].features[j].entry=static_cast<T>(0);
			return sparse_matrix[i_col].features[j].entry;
		}

		/** load sparse matrix from file
		 *
		 * @param loader File object via which to load data
		 */
		void load(CFile* loader);

		/** load sparse matrix from libsvm file together with labels
		 *
		 * @param libsvm_file the libsvm file
		 * @param do_sort_features whether to sort the vector indices (such that they are in
		 * ascending order) after loading
		 * @return label vector
		 */
		SGVector<float64_t> load_with_labels(CLibSVMFile* libsvm_file, bool do_sort_features=true);

		/** save sparse matrix to file
		 *
		 * @param saver File object via which to save data
		 */
		void save(CFile* saver);

		/** save sparse matrix together with labels to file
		 *
		 * @param saver File object via which to save data
		 * @param labels label vector
		 */
		void save_with_labels(CLibSVMFile* saver, SGVector<float64_t> labels);

		/** return the transposed of the sparse matrix */
		SGSparseMatrix<T> get_transposed();

		/** create a sparse matrix from a dense one
		 *
		 * @param full the dense matrix to create the sparse one from
		 */
		void from_dense(SGMatrix<T> full);

		/** sort the indices of the sparse matrix such that they are in ascending order */
		void sort_features();

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
