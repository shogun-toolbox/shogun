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
#include <shogun/lib/memory.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGReferencedData.h>
#include <shogun/lib/SGSparseVector.h>

namespace shogun
{
/** @brief template class SGSparseMatrix */
template <class T> class SGSparseMatrix : public SGReferencedData
{
	public:
		/** default constructor */
		SGSparseMatrix() : SGReferencedData()
		{
			init_data();
		}

		/** constructor for setting params */
		SGSparseMatrix(SGSparseVector<T>* vecs, index_t num_feat,
				index_t num_vec, bool ref_counting=true) :
			SGReferencedData(ref_counting),
			num_vectors(num_vec), num_features(num_feat),
			sparse_matrix(vecs)
		{
		}

		/** constructor to create new matrix in memory */
		SGSparseMatrix(index_t num_vec, index_t num_feat, bool ref_counting=true) :
			SGReferencedData(ref_counting),
			num_vectors(num_vec), num_features(num_feat)
		{
			sparse_matrix=SG_MALLOC(SGSparseVector<T>, num_vectors);
			for (int32_t i=0; i<num_vectors; i++)
			{
				new (&sparse_matrix[i]) SGSparseVector<T>();
				sparse_matrix[i] = SGSparseVector<T>(num_feat);
			}
		}

		/** copy constructor */
		SGSparseMatrix(const SGSparseMatrix &orig) : SGReferencedData(orig)
		{
			copy_data(orig);
		}

		/** destructor */
		virtual ~SGSparseMatrix()
		{
			unref();
		}

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

protected:

		/** copy data */
		virtual void copy_data(const SGReferencedData& orig)
		{
			sparse_matrix = ((SGSparseMatrix*)(&orig))->sparse_matrix;
			num_vectors = ((SGSparseMatrix*)(&orig))->num_vectors;
			num_features = ((SGSparseMatrix*)(&orig))->num_features;
		}

		/** init data */
		virtual void init_data()
		{
			sparse_matrix = NULL;
			num_vectors = 0;
			num_features = 0;
		}

		/** free data */
		virtual void free_data()
		{
			for (int32_t i=0; i<num_vectors; i++)
				(&sparse_matrix[i])->~SGSparseVector<T>();

			SG_FREE(sparse_matrix);
			num_vectors = 0;
			num_features = 0;
		}

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
