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
#include "lib/Set.h"

using namespace shogun;

#ifdef TRACE_MEMORY_ALLOCS
extern CSet<MemoryBlock>* sg_mallocs;
#endif

void* operator new(size_t size) throw (std::bad_alloc)
{
	void *p=malloc(size);
#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
		sg_mallocs->add(MemoryBlock(p,size));
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
	if (sg_mallocs)
		sg_mallocs->remove(MemoryBlock(p));
#endif
	if (p)
		free(p);
}

void* operator new[](size_t size)
{
	void *p=malloc(size);
#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
		sg_mallocs->add(MemoryBlock(p,size));
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
	if (sg_mallocs)
		sg_mallocs->remove(MemoryBlock(p));
#endif
	if (p)
		free(p);
}

#ifdef TRACE_MEMORY_ALLOCS
void list_memory_allocs()
{
	if (sg_mallocs)
	{
		int32_t num=sg_mallocs->get_num_elements();
		printf("%d Blocks are allocated:\n", num);

		for (int32_t i=0; i<num; i++)
			sg_mallocs->get_element(i).display();
	}
}
#endif
