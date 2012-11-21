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
#include <shogun/lib/common.h>
#include <shogun/lib/Map.h>
#include <shogun/base/SGObject.h>

using namespace shogun;

#ifdef TRACE_MEMORY_ALLOCS
extern CMap<void*, shogun::MemoryBlock>* sg_mallocs;

MemoryBlock::MemoryBlock() : ptr(NULL), size(0), file(NULL),
	line(-1), is_sgobject(false)
{
}

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
		sg_mallocs->add(p, MemoryBlock(p,size));
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

void operator delete(void *p) throw()
{
#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
		sg_mallocs->remove(p);
#endif
	free(p);
}

void* operator new[](size_t size) throw(std::bad_alloc)
{
	void *p=malloc(size);
#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
		sg_mallocs->add(p, MemoryBlock(p,size));
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

void operator delete[](void *p) throw()
{
#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
		sg_mallocs->remove(p);
#endif
	free(p);
}

namespace shogun
{
void* sg_malloc(size_t size
#ifdef TRACE_MEMORY_ALLOCS
		, const char* file, int line
#endif
)
{
	void* p=malloc(size);
#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
		sg_mallocs->add(p, MemoryBlock(p,size, file, line));
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
		sg_mallocs->add(p, MemoryBlock(p,size, file, line));
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
		sg_mallocs->remove(ptr);
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
		sg_mallocs->remove(ptr);

	if (sg_mallocs)
		sg_mallocs->add(p, MemoryBlock(p,size, file, line));
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
	MemoryBlock* temp;
	if (sg_mallocs)
	{
		int32_t num=sg_mallocs->get_num_elements();
		int32_t size=sg_mallocs->get_array_size();
		printf("%d Blocks are allocated:\n", num);


		for (int32_t i=0; i<size; i++)
		{
			temp=sg_mallocs->get_element_ptr(i);
			if (temp!=NULL)			
				temp->display();
		}
	}
}
#endif

#ifdef TRACE_MEMORY_ALLOCS
#define SG_SPECIALIZED_MALLOC(type)																\
template<> type* sg_generic_malloc<type >(size_t len, const char* file, int line)				\
{																								\
	return new type[len]();																		\
}																								\
																								\
template<> type* sg_generic_calloc<type >(size_t len, const char* file, int line)				\
{																								\
	return new type[len]();																		\
}																								\
																								\
template<> type* sg_generic_realloc<type >(type* ptr, size_t old_len, size_t len, const char* file, int line)	\
{																								\
	type* new_ptr = new type[len]();															\
	size_t min_len=old_len;																		\
	if (len<min_len)																			\
		min_len=len;																			\
	for (size_t i=0; i<min_len; i++)															\
		new_ptr[i]=ptr[i];																		\
	delete[] ptr;																				\
	return new_ptr;																				\
}																								\
																								\
template<> void sg_generic_free<type >(type* ptr)												\
{																								\
	delete[] ptr;																				\
}

#else // TRACE_MEMORY_ALLOCS

#define SG_SPECIALIZED_MALLOC(type)									\
template<> type* sg_generic_malloc<type >(size_t len)				\
{																	\
	return new type[len]();											\
}																	\
																	\
template<> type* sg_generic_calloc<type >(size_t len)				\
{																	\
	return new type[len]();											\
}																	\
																	\
template<> type* sg_generic_realloc<type >(type* ptr, size_t old_len, size_t len)	\
{																	\
	type* new_ptr = new type[len]();								\
	size_t min_len=old_len;											\
	if (len<min_len)												\
		min_len=len;												\
	for (size_t i=0; i<min_len; i++)								\
		new_ptr[i]=ptr[i];											\
	delete[] ptr;													\
	return new_ptr;													\
}																	\
																	\
template<> void sg_generic_free<type >(type* ptr)					\
{																	\
	delete[] ptr;													\
}
#endif // TRACE_MEMORY_ALLOCS

SG_SPECIALIZED_MALLOC(SGVector<bool>)
SG_SPECIALIZED_MALLOC(SGVector<char>)
SG_SPECIALIZED_MALLOC(SGVector<int8_t>)
SG_SPECIALIZED_MALLOC(SGVector<uint8_t>)
SG_SPECIALIZED_MALLOC(SGVector<int16_t>)
SG_SPECIALIZED_MALLOC(SGVector<uint16_t>)
SG_SPECIALIZED_MALLOC(SGVector<int32_t>)
SG_SPECIALIZED_MALLOC(SGVector<uint32_t>)
SG_SPECIALIZED_MALLOC(SGVector<int64_t>)
SG_SPECIALIZED_MALLOC(SGVector<uint64_t>)
SG_SPECIALIZED_MALLOC(SGVector<float32_t>)
SG_SPECIALIZED_MALLOC(SGVector<float64_t>)
SG_SPECIALIZED_MALLOC(SGVector<floatmax_t>)
#undef SG_SPECIALIZED_MALLOC
}
