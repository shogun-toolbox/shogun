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
		vec_index(0), num_feat_entries(0), features(NULL), do_free(false) {}

	/** constructor for setting params */
	SGSparseVector(SGSparseVectorEntry<T>* feats, index_t num_entries,
			index_t index, bool free_v=false) :
			vec_index(index), num_feat_entries(num_entries), features(feats),
			do_free(free_v) {}

	/** constructor to create new vector in memory */
	SGSparseVector(index_t num_entries, index_t index, bool free_v=false) :
		vec_index(index), num_feat_entries(num_entries), do_free(free_v)
	{
		features=SG_MALLOC(SGSparseVectorEntry<T>, num_feat_entries);
	}

	/** copy constructor */
	SGSparseVector(const SGSparseVector& orig) :
			vec_index(orig.vec_index), num_feat_entries(orig.num_feat_entries),
			features(orig.features), do_free(orig.do_free) {}

	/** free vector */
	void free_vector()
	{
		if (do_free)
			SG_FREE(features);

		features=NULL;
		do_free=false;
		vec_index=0;
		num_feat_entries=0;
	}

	/** destroy vector */
	void destroy_vector()
	{
		do_free=true;
		free_vector();
	}

public:
	/** vector index */
	index_t vec_index;

	/** number of feature entries */
	index_t num_feat_entries;

	/** features */
	SGSparseVectorEntry<T>* features;

	/** whether vector needs to be freed */
	bool do_free;
};
}
#endif // __SGSPARSEVECTOR_H__
