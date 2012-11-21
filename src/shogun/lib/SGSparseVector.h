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

#ifndef __SGSPARSEVECTOR_H__
#define __SGSPARSEVECTOR_H__

#include <shogun/lib/config.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGReferencedData.h>
#include <map>

namespace shogun
{


/** @brief template class SGSparseVectorEntry */
template <class T> struct SGSparseVectorEntry
{
	/** feature index  */
	index_t feat_index;
	/** entry ... */
	T entry;
};

/** @brief template class SGSparseVector
 * The assumtion is that the stored SGSparseVectorEntry<T>* vector is ordered
 * by SGSparseVectorEntry.feat_index in non-decreasing order.
 * This has to be assured by the user of the class.
 */
template <class T> class SGSparseVector : public SGReferencedData
{
public:
	/** default constructor */
	SGSparseVector();

	/** constructor for setting params
	 *
	 * @param feats vector of SGSparseVectorEntry ordered by SGSparseVectorEntry.feat_index in non-decreasing order
	 * @param num_entries number of elements in feats vector
	 * @param ref_counting use reference counting
	 */
	SGSparseVector(SGSparseVectorEntry<T>* feats, index_t num_entries,
			bool ref_counting=true);

	/** constructor to create new vector in memory */
	SGSparseVector(index_t num_entries, bool ref_counting=true);

	/** copy constructor */
	SGSparseVector(const SGSparseVector& orig);

	virtual ~SGSparseVector();

	/** compute the dot product between dense weights and a sparse feature vector
	 * alpha * sparse^T * w + b
	 *
	 * possible with subset
	 *
	 * @param alpha scalar to multiply with
	 * @param vec dense vector to compute dot product with
	 * @param dim length of the dense vector
	 * @param b bias
	 * @return dot product between dense weights and a sparse feature vector
	 */
	T dense_dot(T alpha, T* vec, int32_t dim, T b);

	/** compute the dot product between current sparse vector and a given
	 * sparse vector.
	 * sparse_a^T * sparse_b
	 *
	 * @param v sparse vector
	 * @return dot product between the current sparse vector and v sparse vector
	 */
	T sparse_dot(const SGSparseVector<T>& v);

	/** compute the dot product between two sparse vectors.
	 * sparse_a^T * sparse_b
	 *
	 * @param a sparse vector
	 * @param b sparse vector
	 * @return dot product between a and b
	 */
	static T sparse_dot(const SGSparseVector<T>& a, const SGSparseVector<T>& b);
protected:

	virtual void copy_data(const SGReferencedData& orig);

	virtual void init_data();

	virtual void free_data();

	static int32_t floor_log(index_t n);

	static int32_t cmp_dot_prod_symmetry_fast(index_t alen, index_t blen);

	static T dot_prod_asymmetric(const SGSparseVector<T>& a, const SGSparseVector<T>& b);

	static T dot_prod_symmetric(const SGSparseVector<T>& a, const SGSparseVector<T>& b);

public:
	/** number of feature entries */
	index_t num_feat_entries;

	/** features */
	SGSparseVectorEntry<T>* features;

};

}

#endif // __SGSPARSEVECTOR_H__
