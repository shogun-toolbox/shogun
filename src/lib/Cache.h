#ifndef _CACHE_H__
#define _CACHE_H__

#include "lib/Mathmatics.h"
#include "lib/io.h"
#include <assert.h>
#include <stdlib.h>

template<class T> class CCache
{
	struct TEntry
	{
		long usage_count;
		bool locked;
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
		if (cache_size==NULL || obj_size==NULL || num_entries==NULL)
		{
			CIO::message("doing without cache.\n");
			cache_block=NULL;
			lookup_table=NULL;
			cache_table=NULL;
			cache_is_full=false;
			nr_cache_lines=0;
			entry_size=0;
			return;
		}

		entry_size=obj_size;
		nr_cache_lines=math.min(cache_size*1024*1024/obj_size/sizeof(T), num_entries+1);

		CIO::message("creating %d cache lines (total size: %ld byte)\n", nr_cache_lines, nr_cache_lines*obj_size*sizeof(T));
		cache_block=new T[obj_size*nr_cache_lines];
		lookup_table=new TEntry[num_entries];
		cache_table=new TEntry*[nr_cache_lines];

		assert(cache_block != NULL);
		assert(lookup_table != NULL);
		assert(cache_table != NULL);

		long i;
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
	}

	~CCache()
	{
		delete[] cache_block;
		delete[] lookup_table;
		delete[] cache_table;
	}

	/// returns a cache entry or NULL when not cached
	inline T* lock_entry(long number)
	{
		if (lookup_table)
		{
			//CIO::message("G:%5d: %5d %10d\n", lookup_table[number].usage_count, number, lookup_table[number].obj);
			lookup_table[number].usage_count++;
			lookup_table[number].locked=true;
			return lookup_table[number].obj;
		}
		else
			return NULL;
	}
	
	/// unlocks a cache entry
	inline void unlock_entry(long number)
	{
		if (lookup_table)
			lookup_table[number].locked=false;
	}

	/// returns the address of a free cache entry
	/// to where the data of size obj_size has to
	/// be written
	T* set_entry(long number)
	{
		if (lookup_table)
		{
			// first look for the element with smallest usage count
			//long min_idx=((nr_cache_lines-1)*rand())/(RAND_MAX+1); //avoid the last elem and the scratch line
			long min_idx=0;
			long min=-1;
			bool found_free_line=false;

			if (cache_table[min_idx])
			{
				min=cache_table[min_idx]->usage_count;

				for (long i=0; i<nr_cache_lines; i++)
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
						long v=cache_table[i]->usage_count;

						if (v<min && !cache_table[i]->locked)
						{
							min=v;
							min_idx=i;
							found_free_line=true;
						}
					}
				}
			}
			else
				found_free_line=true;

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
				//CIO::message("S:%5d: %5d(Y)\n", min, number);

				return lookup_table[number].obj;
			}
			else
				return NULL;
		}
		else 
			return NULL;
	}

	bool cache_is_full;
	long entry_size;
	long nr_cache_lines;
	TEntry* lookup_table;
	TEntry** cache_table;
	T* cache_block;
};
#endif
