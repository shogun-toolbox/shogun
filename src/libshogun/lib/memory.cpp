/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <stdio.h>

#include "lib/ShogunException.h"
#include "lib/memory.h"
#include "lib/common.h"

#ifdef TRACE_MEMORY_ALLOCS

class CMemoryBlock
{
	public:
		CMemoryBlock(void* p)
		{
			ptr=p;
			sz=0;
		}

		CMemoryBlock(void* p, size_t sz)
		{
			ptr=p;
			sz=size;
		}

        CMemoryBlock(const CMemoryBlock &b)
        {
			ptr=b.ptr;
			size=b.size;
        }


		bool operator==(const CMemoryBlock &b) const
		{
			return ptr==b.ptr;
		}

		void display()
		{
			SG_PRINT("Object at %p of size %lld bytes\n", ptr, (long long int) size);
		}

	protected:
		void* ptr;
		size_t size;
};

#include "lib/Set.h"

CSet<S_MEMORY_BLOCK> memory_allocs;
#endif

void* operator new(size_t size) throw (std::bad_alloc)
{
	void *p=malloc(size);
#ifdef TRACE_MEMORY_ALLOCS
	memory_allocs.add(CMemoryBlock(p,size));
#endif
	if (!p)
	{
		const size_t buf_len=128;
		char buf[buf_len];
		size_t written=snprintf(buf, buf_len,
			"Out of memory error, tried to allocate %lld bytes using new().\n", (long long int) size);
		if (written<buf_len)
			throw ShogunException(buf);
		else
			throw ShogunException("Out of memory error using new.\n");
	}

	return p;
}

void operator delete(void *p)
{
#ifdef TRACE_MEMORY_ALLOCS
	memory_allocs.remove(CMemoryBlock(p));
#endif
	if (p)
		free(p);
}

void* operator new[](size_t size)
{
	void *p=malloc(size);
#ifdef TRACE_MEMORY_ALLOCS
	memory_allocs.add(CMemoryBlock(p,size));
#endif

	if (!p)
	{
		const size_t buf_len=128;
		char buf[buf_len];
		size_t written=snprintf(buf, buf_len,
			"Out of memory error, tried to allocate %lld bytes using new[].\n", (long long int) size);
		if (written<buf_len)
			throw ShogunException(buf);
		else
			throw ShogunException("Out of memory error using new[].\n");
	}

	return p;
}

void operator delete[](void *p)
{
#ifdef TRACE_MEMORY_ALLOCS
	memory_allocs.remove(CMemoryBlock(p));
#endif
	if (p)
		free(p);
}

#ifdef TRACE_MEMORY_ALLOCS
void list_memory_allocs()
{
	int32_t num=memory_allocs.get_num_elements();

	for (int32_t i=0; i<num; i++)
		memory_allocs[i].display();
}
#endif
