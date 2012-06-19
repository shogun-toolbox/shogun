/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Written (W) 2012 Christian Widmer
 * Written (W) 2010,2012 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 * Copyright (C) 2012 Soeren Sonnenburg
 */

#ifndef __SGSPARSEVECTOR_H__
#define __SGSPARSEVECTOR_H__

#include <shogun/lib/config.h>
#include <shogun/lib/DataType.h>
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

/** @brief template class SGSparseVector */
template <class T> class SGSparseVector
{
public:
	/** default constructor */
	SGSparseVector() :
		num_feat_entries(0), features(NULL), do_free(false) {}

	/** constructor for setting params */
	SGSparseVector(SGSparseVectorEntry<T>* feats, index_t num_entries,
			index_t index, bool free_v=false) :
			num_feat_entries(num_entries), features(feats),
			do_free(free_v)
	{
		create_idx_map();
	}

	/** constructor to create new vector in memory */
	SGSparseVector(index_t num_entries, index_t index, bool free_v=false) :
		num_feat_entries(num_entries), do_free(free_v)
	{
		features=SG_MALLOC(SGSparseVectorEntry<T>, num_feat_entries);
	}

	/** copy constructor */
	SGSparseVector(const SGSparseVector& orig) :
			num_feat_entries(orig.num_feat_entries),
			features(orig.features), do_free(orig.do_free)
	{
		create_idx_map();
	}

	/** free vector */
	void free_vector()
	{
		if (do_free)
			SG_FREE(features);

		dense_to_sparse_idx.clear();
		features=NULL;
		do_free=false;
		num_feat_entries=0;
	}

	/** destroy vector */
	void destroy_vector()
	{
		do_free=true;
		free_vector();
	}

	/** create mapping from dense idx to sparse idx */
	void create_idx_map()
	{
		dense_to_sparse_idx.clear();
		for (int32_t i=0; i!=num_feat_entries; i++)
		{
			dense_to_sparse_idx[features[i].feat_index] = i;
		}
	}

	/** operator overload for vector read only access
	 *
	 * @param index dimension to access
	 *
	 */
	inline const T& operator[](index_t index) const
	{
		// lookup complexity is O(log n)
		std::map<index_t, index_t>::const_iterator it = dense_to_sparse_idx.find(index);

		if (it != dense_to_sparse_idx.end())
		{
			// use mapping for lookup
			return features[it->second].entry;
		} else {
			return zero;
		}
	}

		
	/** TODO: operator overload for vector r/w access
	 *
	 * @param index dimension to access
	 *
	inline T& operator[](index_t index)
	{
		return dense_to_sparse_idx[index];
		// lookup complexity is O(log n)
		typename std::map<index_t, T>::iterator it = dense_to_sparse_idx.find(index);

		if (it != dense_to_sparse_idx.end())
		{
			return it->second;
		} else {
			return dense_to_sparse_idx.insert(index, 0);
		}
	}
		*/

public:
	/** number of feature entries */
	index_t num_feat_entries;

	/** features */
	SGSparseVectorEntry<T>* features;

	/** whether vector needs to be freed */
	bool do_free;

protected:	
	/** store mapping of indices */
	std::map<index_t, index_t> dense_to_sparse_idx;

	/** zero element */
	static const T zero;

};

// inititalize static member in template class
template <typename T>
const T SGSparseVector<T>::zero = T(0);

}

#endif // __SGSPARSEVECTOR_H__
