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

/** @brief template class SGSparseVector */
template <class T> class SGSparseVector : public SGReferencedData
{
public:
	/** default constructor */
	SGSparseVector() : SGReferencedData(false)
	{
		init_data();
	}

	/** constructor for setting params */
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

public:
	/** number of feature entries */
	index_t num_feat_entries;

	/** features */
	SGSparseVectorEntry<T>* features;

};

}

#endif // __SGSPARSEVECTOR_H__
