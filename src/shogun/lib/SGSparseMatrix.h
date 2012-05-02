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
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGSparseVector.h>

namespace shogun
{
/** @brief template class SGSparseMatrix */
template <class T> class SGSparseMatrix
{
	public:
		/** default constructor */
		SGSparseMatrix() :
			num_vectors(0), num_features(0), sparse_matrix(NULL),
			do_free(false) { }


		/** constructor for setting params */
		SGSparseMatrix(SGSparseVector<T>* vecs, index_t num_feat,
				index_t num_vec, bool free_m=false) :
			num_vectors(num_vec), num_features(num_feat),
			sparse_matrix(vecs), do_free(free_m) { }

		/** constructor to create new matrix in memory */
		SGSparseMatrix(index_t num_vec, index_t num_feat, bool free_m=false) :
			num_vectors(num_vec), num_features(num_feat), do_free(free_m)
		{
			sparse_matrix=SG_MALLOC(SGSparseVector<T>, num_vectors);
		}

		/** copy constructor */
		SGSparseMatrix(const SGSparseMatrix &orig) :
			num_vectors(orig.num_vectors), num_features(orig.num_features),
			sparse_matrix(orig.sparse_matrix), do_free(orig.do_free) { }

		/** free matrix */
		void free_matrix()
		{
			if (do_free)
				SG_FREE(sparse_matrix);

			sparse_matrix=NULL;
			do_free=false;
			num_vectors=0;
			num_features=0;
		}

		/** own matrix */
		void own_matrix()
		{
			for (index_t i=0; i<num_vectors; i++)
				sparse_matrix[i].do_free=false;

			do_free=false;
		}

		/** destroy matrix */
		void destroy_matrix()
		{
			do_free=true;
			free_matrix();
		}

	public:
	/// total number of vectors
	index_t num_vectors;

	/// total number of features
	index_t num_features;

	/// array of sparse vectors of size num_vectors
	SGSparseVector<T>* sparse_matrix;

	/** whether vector needs to be freed */
	bool do_free;
};
}
#endif // __SGNDARRAY_H__
