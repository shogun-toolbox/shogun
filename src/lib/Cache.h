#ifndef _CACHE_H__
#define _CACHE_H__

#include <assert.h>
#include <stdlib.h>

template<class T> class CCache
{
	struct TEntry
	{
		long usage_count;
		T* obj;
	};

public:
	/// create cache of cache_size Megabytes.
	/// num_entries objects can be cached
	/// whose lookup table of 
	/// size: sizeof(long)*num_entries
	/// must fit into memory
	/// a chunk of size cache_size will be allocated
	CCache(long cache_size, long obj_size, long num_entries)
	{
		entry_size=obj_size;
		nr_cache_lines=cache_size/obj_size;
		cache_block=new T[obj_size*nr_cache_lines];
		lookup_table=new TEntry[num_entries];
		cache_table=new TEntry*[nr_cache_lines];

		assert(cache_block != NULL);
		assert(lookup_table != NULL);
		assert(cache_table != NULL);
	}

	~CCache()
	{
		delete[] cache_block;
		delete[] lookup_table;
		delete[] cache_table;
	}

	/// returns a cache entry or NULL when not cached
	inline T* get_entry(long number)
	{
		lookup_table[number].usage_count++;
		return lookup_table[number].obj;
	}

	/// returns the address of a free cache entry
	/// to where the data of size obj_size has to
	/// be written
	T* set_entry(long number)
	{
		// first look for the element with smallest usage count
		long min_idx=(nr_cache_lines*rand())/(RAND_MAX+1);
		long min=(*cache_table[min_idx]).usage_count;

		for (long i=0; i<nr_cache_lines; i++)
		{
			long v=(*cache_table[i]).usage_count;

			if (v<min)
			{
				min=v;
				min_idx=i;
			}
		}

		// and overwrite it.
		cache_table[min_idx]=&lookup_table[number];

		lookup_table[number].usage_count++;
		lookup_table[number].obj=&cache_block[entry_size*min_idx];

		return lookup_table[number].obj;
	}


	long entry_size;
	long nr_cache_lines;
	T* cache_block;
	TEntry* lookup_table;
	TEntry** cache_table;
};
#endif
