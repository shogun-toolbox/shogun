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
	SGSparseVector() : SGReferencedData()
	{
		init_data();
	}

	/** constructor for setting params
	 *
	 * @param feats vector of SGSparseVectorEntry ordered by SGSparseVectorEntry.feat_index in non-decreasing order
	 * @param num_entries number of elements in feats vector
	 * @param ref_counting use reference counting
	 */
	SGSparseVector(SGSparseVectorEntry<T>* feats, index_t num_entries,
			bool ref_counting=true) :
			SGReferencedData(ref_counting),
			num_feat_entries(num_entries), features(feats)
	{
	}

	/** constructor to create new vector in memory */
	SGSparseVector(index_t num_entries, bool ref_counting=true) :
		SGReferencedData(ref_counting),
		num_feat_entries(num_entries)
	{
		features = SG_MALLOC(SGSparseVectorEntry<T>, num_feat_entries);
	}

	/** copy constructor */
	SGSparseVector(const SGSparseVector& orig) :
		SGReferencedData(orig)
	{
		copy_data(orig);
	}

	virtual ~SGSparseVector()
	{
		unref();
	}

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
	T dense_dot(T alpha, T* vec, int32_t dim, T b)
	{
		ASSERT(vec);
		T result=b;

		if (features)
		{
			for (int32_t i=0; i<num_feat_entries; i++)
			{
				result+=alpha*vec[features[i].feat_index]
					*features[i].entry;
			}
		}

		return result;
	}

	/** compute the dot product between current sparse vector and a given
	 * sparse vector.
	 * sparse_a^T * sparse_b
	 *
	 * @param v sparse vector
	 * @return dot product between the current sparse vector and v sparse vector
	 */
	T sparse_dot(const SGSparseVector<T>& v)
	{
		return sparse_dot(*this, v);
	}

	/** compute the dot product between two sparse vectors.
	 * sparse_a^T * sparse_b
	 *
	 * @param a sparse vector
	 * @param b sparse vector
	 * @return dot product between a and b
	 */
	static T sparse_dot(const SGSparseVector<T>& a, const SGSparseVector<T>& b)
	{
		if (a.num_feat_entries == 0 || b.num_feat_entries == 0)
			return 0;

		int32_t cmp = cmp_dot_prod_symmetry_fast(a.num_feat_entries, b.num_feat_entries);

		if (cmp == 0) // symmetric
		{
			return dot_prod_symmetric(a, b);
		}
		else if (cmp > 0) // b has more element
		{
			return dot_prod_asymmetric(a, b);
		}
		else // a has more element
		{
			return dot_prod_asymmetric(b, a);
		}
	}

protected:

	virtual void copy_data(const SGReferencedData& orig)
	{
		num_feat_entries = ((SGSparseVector*)(&orig))->num_feat_entries;
		features = ((SGSparseVector*)(&orig))->features;
	}

	virtual void init_data()
	{
		num_feat_entries = 0;
		features = NULL;
	}

	virtual void free_data()
	{
		num_feat_entries = 0;
		SG_FREE(features);
	}

	static int32_t floor_log(index_t n)
	{
		register int32_t i;
		for (i = 0; n != 0; i++)
			n >>= 1;

		return i;
	}

	static int32_t cmp_dot_prod_symmetry_fast(index_t alen, index_t blen)
	{
		if (alen > blen) // no need for floats here
		{
			return (blen * floor_log(alen) < alen) ? -1 : 0;
		}
		else // alen <= blen
		{
			return (alen * floor_log(blen) < blen) ? 1 : 0;
		}
	}

	static T dot_prod_asymmetric(const SGSparseVector<T>& a, const SGSparseVector<T>& b)
	{
		T dot_prod = 0;
		for(index_t b_idx = 0; b_idx < b.num_feat_entries; ++b_idx)
		{
			const T tmp = b.features[b_idx].entry;
			if (a.features[a.num_feat_entries-1].feat_index < b.features[b_idx].feat_index)
				break;
			for (index_t a_idx = 0; a_idx < a.num_feat_entries; ++a_idx)
			{
				if (a.features[a_idx].feat_index == b.features[b_idx].feat_index)
					dot_prod += tmp * a.features[a_idx].entry;
			}
		}
		return dot_prod;
	}

	static T dot_prod_symmetric(const SGSparseVector<T>& a, const SGSparseVector<T>& b)
	{
		ASSERT(a.num_feat_entries > 0 && b.num_feat_entries > 0);
		T dot_prod = 0;
		index_t a_idx = 0, b_idx = 0;
		while (true)
		{
			if (a.features[a_idx].feat_index == b.features[b_idx].feat_index)
			{
				dot_prod += a.features[a_idx].entry * b.features[b_idx].entry;

				a_idx++;
				if (a.num_feat_entries == a_idx)
					break;
				b_idx++;
				if (b.num_feat_entries == b_idx)
					break;
			}
			else if (a.features[a_idx].feat_index < b.features[b_idx].feat_index)
			{
				a_idx++;
				if (a.num_feat_entries == a_idx)
					break;
			}
			else
			{
				b_idx++;
				if (b.num_feat_entries == b_idx)
					break;
			}
		}
		return dot_prod;
	}

public:
	/** number of feature entries */
	index_t num_feat_entries;

	/** features */
	SGSparseVectorEntry<T>* features;

};

}

#endif // __SGSPARSEVECTOR_H__
