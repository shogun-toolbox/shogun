/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/ShogunException.h>
#include <shogun/lib/memory.h>
#include <shogun/lib/common.h>
#include <shogun/lib/Set.h>
#include <shogun/base/SGObject.h>

using namespace shogun;

#ifdef TRACE_MEMORY_ALLOCS
extern CSet<shogun::MemoryBlock>* sg_mallocs;

MemoryBlock::MemoryBlock(void* p) : ptr(p), size(0), file(NULL),
	line(-1), is_sgobject(false)
{
}

MemoryBlock::MemoryBlock(void* p, size_t sz, const char* fname, int linenr) :
	ptr(p), size(sz), file(fname), line(linenr), is_sgobject(false)
{
}

MemoryBlock::MemoryBlock(const MemoryBlock &b)
{
	ptr=b.ptr;
	size=b.size;
	file=b.file;
	line=b.line;
	is_sgobject=b.is_sgobject;
}


bool MemoryBlock::operator==(const MemoryBlock &b) const
{
	return ptr==b.ptr;
}

void MemoryBlock::display()
{
	if (line!=-1)
	{
		printf("Memory block at %p of size %lld bytes (allocated in %s line %d)\n",
				ptr, (long long int) size, file, line);
	}
	else
	{
		if (is_sgobject)
		{
			CSGObject* obj=(CSGObject*) ptr;
			printf("SGObject '%s' at %p of size %lld bytes with %d ref's\n",
					obj->get_name(), obj, (long long int) size, obj->ref_count());
		}
		else
		{
			printf("Object at %p of size %lld bytes\n",
					ptr, (long long int) size);
		}
	}
}

void MemoryBlock::set_sgobject()
{
	is_sgobject=true;
}
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
		sg_mallocs->remove(MemoryBlock(p, true));
#endif
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
		sg_mallocs->remove(MemoryBlock(p, false));
#endif
	free(p);
}

void* sg_malloc(size_t size
#ifdef TRACE_MEMORY_ALLOCS
		, const char* file, int line
#endif
)
{
	void* p=malloc(size);
#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
		sg_mallocs->add(MemoryBlock(p,size, file, line));
#endif

	if (!p)
	{
		const size_t buf_len=128;
		char buf[buf_len];
		size_t written=snprintf(buf, buf_len,
			"Out of memory error, tried to allocate %lld bytes using malloc.\n", (long long int) size);
		if (written<buf_len)
			throw ShogunException(buf);
		else
			throw ShogunException("Out of memory error using malloc.\n");
	}

	return p;
}

void* sg_calloc(size_t num, size_t size
#ifdef TRACE_MEMORY_ALLOCS
		, const char* file, int line
#endif
)
{
	void* p=calloc(num, size);
#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
		sg_mallocs->add(MemoryBlock(p,size, file, line));
#endif

	if (!p)
	{
		const size_t buf_len=128;
		char buf[buf_len];
		size_t written=snprintf(buf, buf_len,
			"Out of memory error, tried to allocate %lld bytes using calloc.\n",
			(long long int) size);

		if (written<buf_len)
			throw ShogunException(buf);
		else
			throw ShogunException("Out of memory error using calloc.\n");
	}

	return p;
}

void  sg_free(void* ptr)
{
#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
		sg_mallocs->remove(MemoryBlock(ptr, false));
#endif
	free(ptr);
}

void* sg_realloc(void* ptr, size_t size
#ifdef TRACE_MEMORY_ALLOCS
		, const char* file, int line
#endif
)
{
	void* p=realloc(ptr, size);

#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
		sg_mallocs->remove(MemoryBlock(ptr, false));

	if (sg_mallocs)
		sg_mallocs->add(MemoryBlock(p,size, file, line));
#endif

	if (!p && (size || !ptr))
	{
		const size_t buf_len=128;
		char buf[buf_len];
		size_t written=snprintf(buf, buf_len,
			"Out of memory error, tried to allocate %lld bytes using realloc.\n", (long long int) size);
		if (written<buf_len)
			throw ShogunException(buf);
		else
			throw ShogunException("Out of memory error using realloc.\n");
	}

	return p;
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
