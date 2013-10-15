/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/config.h>
#include <shogun/lib/ShogunException.h>
#include <shogun/lib/common.h>
#include <shogun/lib/Map.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/base/SGObject.h>

#include <string.h>

#ifdef USE_JEMALLOC
#include <jemalloc/jemalloc.h>
#elif USE_TCMALLOC
#include <gperftools/tcmalloc.h>
#endif

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

#ifdef HAVE_CXX11
void* operator new(size_t size)
#else
void* operator new(size_t size) throw (std::bad_alloc)
#endif
{
#if defined(USE_JEMALLOC)
	void *p=je_malloc(size);
#elif defined(USE_TCMALLOC)
	void *p=tc_malloc(size);
#else
	void *p=malloc(size);
#endif

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

#if defined(USE_JEMALLOC)
	je_free(p);
#elif defined(USE_TCMALLOC)
	tc_free(p);
#else
	free(p);
#endif
}

#ifdef HAVE_CXX11
void* operator new[](size_t size)
#else
void* operator new[](size_t size) throw(std::bad_alloc)
#endif
{
#if defined(USE_JEMALLOC)
	void *p=je_malloc(size);
#elif defined(USE_TCMALLOC)
	void *p=tc_malloc(size);
#else
	void *p=malloc(size);
#endif

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

#if defined(USE_JEMALLOC)
	je_free(p);
#elif defined(USE_TCMALLOC)
	tc_free(p);
#else
	free(p);
#endif
}

namespace shogun
{
void* sg_malloc(size_t size
#ifdef TRACE_MEMORY_ALLOCS
		, const char* file, int line
#endif
)
{
#if defined(USE_JEMALLOC)
	void* p=je_malloc(size);
#elif defined(USE_TCMALLOC)
	void *p=tc_malloc(size);
#else
	void* p=malloc(size);
#endif
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
#if defined(USE_JEMALLOC)
	void* p=je_calloc(num, size);
#elif defined(USE_TCMALLOC)
	void* p=tc_calloc(num, size);
#else
	void* p=calloc(num, size);
#endif

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

#if defined(USE_JEMALLOC)
	je_free(ptr);
#elif defined(USE_TCMALLOC)
	tc_free(ptr);
#else
	free(ptr);
#endif
}

void* sg_realloc(void* ptr, size_t size
#ifdef TRACE_MEMORY_ALLOCS
		, const char* file, int line
#endif
)
{
#if defined(USE_JEMALLOC)
	void* p=je_realloc(ptr, size);
#elif defined(USE_TCMALLOC)
	void* p=tc_realloc(ptr, size);
#else
	void* p=realloc(ptr, size);
#endif

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
SG_SPECIALIZED_MALLOC(SGVector<complex128_t>)

SG_SPECIALIZED_MALLOC(SGSparseVector<bool>)
SG_SPECIALIZED_MALLOC(SGSparseVector<char>)
SG_SPECIALIZED_MALLOC(SGSparseVector<int8_t>)
SG_SPECIALIZED_MALLOC(SGSparseVector<uint8_t>)
SG_SPECIALIZED_MALLOC(SGSparseVector<int16_t>)
SG_SPECIALIZED_MALLOC(SGSparseVector<uint16_t>)
SG_SPECIALIZED_MALLOC(SGSparseVector<int32_t>)
SG_SPECIALIZED_MALLOC(SGSparseVector<uint32_t>)
SG_SPECIALIZED_MALLOC(SGSparseVector<int64_t>)
SG_SPECIALIZED_MALLOC(SGSparseVector<uint64_t>)
SG_SPECIALIZED_MALLOC(SGSparseVector<float32_t>)
SG_SPECIALIZED_MALLOC(SGSparseVector<float64_t>)
SG_SPECIALIZED_MALLOC(SGSparseVector<floatmax_t>)
SG_SPECIALIZED_MALLOC(SGSparseVector<complex128_t>)

SG_SPECIALIZED_MALLOC(SGMatrix<bool>)
SG_SPECIALIZED_MALLOC(SGMatrix<char>)
SG_SPECIALIZED_MALLOC(SGMatrix<int8_t>)
SG_SPECIALIZED_MALLOC(SGMatrix<uint8_t>)
SG_SPECIALIZED_MALLOC(SGMatrix<int16_t>)
SG_SPECIALIZED_MALLOC(SGMatrix<uint16_t>)
SG_SPECIALIZED_MALLOC(SGMatrix<int32_t>)
SG_SPECIALIZED_MALLOC(SGMatrix<uint32_t>)
SG_SPECIALIZED_MALLOC(SGMatrix<int64_t>)
SG_SPECIALIZED_MALLOC(SGMatrix<uint64_t>)
SG_SPECIALIZED_MALLOC(SGMatrix<float32_t>)
SG_SPECIALIZED_MALLOC(SGMatrix<float64_t>)
SG_SPECIALIZED_MALLOC(SGMatrix<floatmax_t>)
SG_SPECIALIZED_MALLOC(SGMatrix<complex128_t>)
#undef SG_SPECIALIZED_MALLOC
}

void* shogun::get_copy(void* src, size_t len)
{
	void* copy=SG_MALLOC(uint8_t, len);
	memcpy(copy, src, len);
	return copy;
}

char* shogun::get_strdup(const char* str)
{
	if (!str)
		return NULL;

	return (char*) get_copy((void*) str, strlen(str)+1);
}
