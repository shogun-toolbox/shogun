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
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGReferencedData.h>
#include <shogun/io/LibSVMFile.h>

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

		/** operator overload for sparse-matrix read only access
		 * @param i_row
		 * @param i_col
		 */
		inline const T operator()(index_t i_row, index_t i_col) const
		{
			REQUIRE(i_row>=0, "index %d negative!\n", i_row);
			REQUIRE(i_col>=0, "index %d negative!\n", i_col);
			REQUIRE(i_row<num_vectors, "index should be less than %d, %d provided!\n",
				num_vectors, i_row);
			REQUIRE(i_col<num_features, "index should be less than %d, %d provided!\n",
				num_features, i_col);

			for (index_t i=0; i<sparse_matrix[i_row].num_feat_entries; ++i)
			{
				if (i_col==sparse_matrix[i_row].features[i].feat_index)
					return sparse_matrix[i_row].features[i].entry;
			}
			return 0;
		}

		/** operator overload for sparse-matrix r/w access
		 * @param i_row
		 * @param i_col
		 */
		inline T& operator()(index_t i_row, index_t i_col)
		{
			REQUIRE(i_row>=0, "index %d negative!\n", i_row);
			REQUIRE(i_col>=0, "index %d negative!\n", i_col);
			REQUIRE(i_row<num_vectors, "index should be less than %d, %d provided!\n",
				num_vectors, i_row);
			REQUIRE(i_col<num_features, "index should be less than %d, %d provided!\n",
				num_features, i_col);

			for (index_t i=0; i<sparse_matrix[i_row].num_feat_entries; ++i)
			{
				if (i_col==sparse_matrix[i_row].features[i].feat_index)
					return sparse_matrix[i_row].features[i].entry;
			}
			index_t j=sparse_matrix[i_row].num_feat_entries;
			sparse_matrix[i_row].num_feat_entries=j+1;
			sparse_matrix[i_row].features=SG_REALLOC(SGSparseVectorEntry<T>,
				sparse_matrix[i_row].features, j, j+1);
			sparse_matrix[i_row].features[j].feat_index=i_col;
			sparse_matrix[i_row].features[j].entry=static_cast<T>(0);
			
			return sparse_matrix[i_row].features[j].entry;
		}

		/** load sparse matrix from file
		 *
		 * @param loader File object via which to load data
		 */
		void load(CFile* loader);

		/** TODO add comment */
		SGSparseMatrix<T> get_transposed();

		/** TODO add comment */
		CRegressionLabels* load_svmlight_file(CLibSVMFile* file, bool do_sort_features);

		/** TODO add comment */
		void write_svmlight_file(CLibSVMFile* file, CRegressionLabels* labels);

		/** TODO add comment */
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
