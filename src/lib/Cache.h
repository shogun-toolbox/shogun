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
		assert(cache_size !=NULL);
		assert(obj_size !=NULL);
		assert(num_entries !=NULL);

		entry_size=obj_size;
		nr_cache_lines=cache_size*1024*1024/obj_size;
		cache_block=new T[obj_size*nr_cache_lines];
		lookup_table=new TEntry[num_entries];
		cache_table=new TEntry*[nr_cache_lines];

		assert(cache_block != NULL);
		assert(lookup_table != NULL);
		assert(cache_table != NULL);

		long i;
		for (i=0; i<nr_cache_lines; i++)
			cache_table[i]=&lookup_table[i];

		for (i=0; i<num_entries; i++)
		{
			lookup_table[i].usage_count=-1;
			lookup_table[i].obj=NULL;
		}
		cache_is_full=false;

		min=-1;
		min_idx=0;
		
		//reserve the very last cache line
		//as scratch buffer
		nr_cache_lines--;
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
		//CIO::message("G:%5d: %5d\n", lookup_table[number].usage_count, number);
		lookup_table[number].usage_count++;
		if (lookup_table[number].usage_count>min)
		{
			min=lookup_table[number].usage_count;
			min_idx=number;
		}
		return lookup_table[number].obj;
	}

	/// returns the address of a free cache entry
	/// to where the data of size obj_size has to
	/// be written
	T* set_entry(long number)
	{
		// first look for the element with smallest usage count
		long s_min_idx=((nr_cache_lines-1)*rand())/(RAND_MAX+1); //avoid the last elem and the scratch line
		long s_min=(*cache_table[min_idx]).usage_count;

		for (long i=0; i<nr_cache_lines; i++)
		{
			long v=(*cache_table[i]).usage_count;

			if (v<s_min && min_idx!=i)
			{
				s_min=v;
				s_min_idx=i;
			}
		}

		if (s_min_idx==nr_cache_lines-1 || min_idx==nr_cache_lines-1) //since this is an indicator for a full cache
			cache_is_full=true;

		if (  (s_min-min) < 5 && cache_is_full)
			number=nr_cache_lines;
		else
		{
			min_idx=s_min_idx;
			min=s_min;
		}

		// and overwrite it.
		CIO::message("S:%5d: %5d(*)\n", min, number);
		(*cache_table[min_idx]).obj=NULL;
		cache_table[min_idx]=&lookup_table[number];

		lookup_table[number].obj=&cache_block[entry_size*min_idx];
		return lookup_table[number].obj;
	}

	bool cache_is_full;

	long min;
	long min_idx;

	long entry_size;
	long nr_cache_lines;
	T* cache_block;
	TEntry* lookup_table;
	TEntry** cache_table;
};
#endif
