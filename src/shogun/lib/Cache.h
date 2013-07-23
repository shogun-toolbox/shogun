/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CACHE_H__
#define _CACHE_H__

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/SGObject.h>

#include <stdlib.h>

namespace shogun
{
/** @brief Template class Cache implements a simple cache.
 *
 * When the cache is full -- elements that are least used are freed from the
 * cache. Thus for the cache to be effective one should not visit loop over
 * objects, i.e. visit elements in order 0...num_elements (with num_elements >>
 * the maximal number of entries in cache)
 *
 */
template<class T> class CCache : public CSGObject
{
	/** cache entry */
	struct TEntry
	{
		/** usage count */
		int64_t usage_count;
		/** if entry is locked */
		bool locked;
		/** cached object */
		T* obj;
	};

	public:
	 /** default constructor  */
	CCache() :CSGObject()
	{
		SG_UNSTABLE("CCache::CCache()", "\n")

		cache_block=NULL;
		lookup_table=NULL;
		cache_table=NULL;
		cache_is_full=false;
		nr_cache_lines=0;
		entry_size=0;

		set_generic<T>();
	}

	/** constructor
	 *
	 * create a cache in which num_entries objects can be cached
	 * whose lookup table of sizeof(int64_t)*num_entries
	 * must fit into memory
	 *
	 * @param cache_size cache size in Megabytes
	 * @param obj_size object size
	 * @param num_entries number of cached objects
	 */
	CCache(int64_t cache_size, int64_t obj_size, int64_t num_entries)
	: CSGObject()
	{
		if (cache_size==0 || obj_size==0 || num_entries==0)
		{
			SG_INFO("doing without cache.\n")
			cache_block=NULL;
			lookup_table=NULL;
			cache_table=NULL;
			cache_is_full=false;
			nr_cache_lines=0;
			entry_size=0;
			return;
		}

		entry_size=obj_size;
		nr_cache_lines=CMath::min((int64_t) (cache_size*1024*1024/obj_size/sizeof(T)), num_entries+1);

		SG_INFO("creating %d cache lines (total size: %ld byte)\n", nr_cache_lines, nr_cache_lines*obj_size*sizeof(T))
		cache_block=SG_MALLOC(T, obj_size*nr_cache_lines);
		lookup_table=SG_MALLOC(TEntry, num_entries);
		cache_table=SG_MALLOC(TEntry*, nr_cache_lines);

		ASSERT(cache_block)
		ASSERT(lookup_table)
		ASSERT(cache_table)

		int64_t i;
		for (i=0; i<nr_cache_lines; i++)
			cache_table[i]=NULL;

		for (i=0; i<num_entries; i++)
		{
			lookup_table[i].usage_count=-1;
			lookup_table[i].locked=false;
			lookup_table[i].obj=NULL;
		}
		cache_is_full=false;

		//reserve the very last cache line
		//as scratch buffer
		nr_cache_lines--;

		set_generic<T>();
	}

	virtual ~CCache()
	{
		SG_FREE(cache_block);
		SG_FREE(lookup_table);
		SG_FREE(cache_table);
	}

	/** checks if an object is cached
	 *
	 * @param number number of object to check for
	 * @return if an object is cached
	 */
	inline bool is_cached(int64_t number)
	{
		return (lookup_table && lookup_table[number].obj);
	}

	/** lock and get a cache entry
	 *
	 * @param number number of object to lock and get
	 * @return cache entry or NULL when not cached
	 */
	inline T* lock_entry(int64_t number)
	{
		if (lookup_table)
		{
			lookup_table[number].usage_count++;
			lookup_table[number].locked=true;
			return lookup_table[number].obj;
		}
		else
			return NULL;
	}

	/** unlock a cache entry
	 *
	 * @param number number of object to unlock
	 */
	inline void unlock_entry(int64_t number)
	{
		if (lookup_table)
			lookup_table[number].locked=false;
	}

	/** returns the address of a free cache entry
	 * to where the data of size obj_size has to
	 * be written
	 *
	 * @param number number of object to unlock
	 * @return address of a free cache entry
	 */
	T* set_entry(int64_t number)
	{
		if (lookup_table)
		{
			// first look for the element with smallest usage count
			int64_t min_idx=0;
			int64_t min=-1;
			bool found_free_line=false;

			int64_t start=0;
			for (start=0; start<nr_cache_lines; start++)
			{
				if (!cache_table[start])
				{
					min_idx=start;
					min=-1;
					found_free_line=true;
					break;
				}
				else
				{
					if (!cache_table[start]->locked)
					{
						min=cache_table[start]->usage_count;
						min_idx=start;
						found_free_line=true;
						break;
					}
				}
			}

			for (int64_t i=start; i<nr_cache_lines; i++)
			{
				if (!cache_table[i])
				{
					min_idx=i;
					min=-1;
					found_free_line=true;
					break;
				}
				else
				{
					int64_t v=cache_table[i]->usage_count;

					if (v<min && !cache_table[i]->locked)
					{
						min=v;
						min_idx=i;
						found_free_line=true;
					}
				}
			}

			if (cache_table[nr_cache_lines-1]) //since this is an indicator for a full cache
				cache_is_full=true;

			if (found_free_line)
			{
				// and overwrite it.
				if ( (lookup_table[number].usage_count-min) < 5 && cache_is_full && ! (cache_table[nr_cache_lines] && cache_table[nr_cache_lines]->locked))
					min_idx=nr_cache_lines; //scratch entry

				if (cache_table[min_idx])
					cache_table[min_idx]->obj=NULL;

				cache_table[min_idx]=&lookup_table[number];
				lookup_table[number].obj=&cache_block[entry_size*min_idx];

				//lock cache entry;
				lookup_table[number].usage_count=0;
				lookup_table[number].locked=true;
				return lookup_table[number].obj;
			}
			else
				return NULL;
		}
		else
			return NULL;
	}

	/** @return object name */
	virtual const char* get_name() const { return "Cache"; }

	protected:
	/** if cache is full */
	bool cache_is_full;
	/** size of one entry */
	int64_t entry_size;
	/** number of cache lines */
	int64_t nr_cache_lines;
	/** lookup table */
	TEntry* lookup_table;
	/** cache table containing cached objects */
	TEntry** cache_table;
	/** cache block */
	T* cache_block;
};
}
#endif
