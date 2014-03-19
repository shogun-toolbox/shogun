/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __MEMORY_H__
#define __MEMORY_H__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <new>

/* wrappers for malloc, free, realloc, calloc */

/* overload new() / delete */
void* operator new(size_t size) throw (std::bad_alloc);
void operator delete(void *p) throw();

/* overload new[] / delete[] */
void* operator new[](size_t size) throw(std::bad_alloc);
void operator delete[](void *p) throw();

#ifdef TRACE_MEMORY_ALLOCS
#define SG_MALLOC(type, len) sg_generic_malloc<type>(size_t(len), __FILE__, __LINE__)
#define SG_CALLOC(type, len) sg_generic_calloc<type>(size_t(len), __FILE__, __LINE__)
#define SG_REALLOC(type, ptr, old_len, len) sg_generic_realloc<type>(ptr, size_t(old_len), size_t(len), __FILE__, __LINE__)
#define SG_FREE(ptr) sg_generic_free(ptr)
#else //TRACE_MEMORY_ALLOCS

#define SG_MALLOC(type, len) sg_generic_malloc<type>(size_t(len))
#define SG_CALLOC(type, len) sg_generic_calloc<type>(size_t(len))
#define SG_REALLOC(type, ptr, old_len, len) sg_generic_realloc<type>(ptr, size_t(old_len), size_t(len))
#define SG_FREE(ptr) sg_generic_free(ptr)
#endif //TRACE_MEMORY_ALLOCS

namespace shogun
{
	template <class T> class SGVector;
	template <class T> class SGSparseVector;
	template <class T> class SGMatrix;

#ifdef TRACE_MEMORY_ALLOCS
void* sg_malloc(size_t size, const char* file, int line);
template <class T> T* sg_generic_malloc(size_t len, const char* file, int line)
{
	return (T*) sg_malloc(sizeof(T)*len, file, line);
}

void* sg_calloc(size_t num, size_t size, const char* file, int line);
template <class T> T* sg_generic_calloc(size_t len, const char* file, int line)
{
	return (T*) sg_calloc(len, sizeof(T), file, line);
}

void* sg_realloc(void* ptr, size_t size, const char* file, int line);
template <class T> T* sg_generic_realloc(T* ptr, size_t old_len, size_t len, const char* file, int line)
{
	return (T*) sg_realloc(ptr, sizeof(T)*len, file, line);
}

void  sg_free(void* ptr);
template <class T> void sg_generic_free(T* ptr)
{
	sg_free((void*) ptr);
}
#else //TRACE_MEMORY_ALLOCS
void* sg_malloc(size_t size);
template <class T> T* sg_generic_malloc(size_t len)
{
	return (T*) sg_malloc(sizeof(T)*len);
}

void* sg_realloc(void* ptr, size_t size);
template <class T> T* sg_generic_realloc(T* ptr, size_t old_len, size_t len)
{
	return (T*) sg_realloc(ptr, sizeof(T)*len);
}

void* sg_calloc(size_t num, size_t size);
template <class T> T* sg_generic_calloc(size_t len)
{
	return (T*) sg_calloc(len, sizeof(T));
}

void  sg_free(void* ptr);
template <class T> void sg_generic_free(T* ptr)
{
	sg_free(ptr);
}
#endif //TRACE_MEMORY_ALLOCS
#ifdef TRACE_MEMORY_ALLOCS
/** @brief memory block */
class MemoryBlock
{
	public:
		/** default constructor
		 */
		MemoryBlock();
		/** constructor
		 * @param p p
		 */
		MemoryBlock(void* p);
		/** constructor
		 * @param p p
		 * @param sz sz
		 * @param fname fname
		 * @param linenr line number
		 */
		MemoryBlock(void* p, size_t sz, const char* fname=NULL, int linenr=-1);
		/** copy constructor
		 * @param b b
		 */
	MemoryBlock(const MemoryBlock &b);

		/** equality
		 * @param b b
		 */
		bool operator==(const MemoryBlock &b) const;
		/** display */
		void display();
		/** set sg object */
		void set_sgobject();

	protected:
		void* ptr;
		size_t size;
		const char* file;
		int line;
		bool is_sgobject;
};
void list_memory_allocs();
#endif

#ifdef TRACE_MEMORY_ALLOCS
#define SG_SPECIALIZED_MALLOC(type)																\
template<> type* sg_generic_malloc<type >(size_t len, const char* file, int line);				\
template<> type* sg_generic_calloc<type >(size_t len, const char* file, int line);				\
template<> type* sg_generic_realloc<type >(type* ptr, size_t old_len, size_t len, const char* file, int line);	\
template<> void sg_generic_free<type >(type* ptr);
#else // TRACE_MEMORY_ALLOCS
#define SG_SPECIALIZED_MALLOC(type)													\
template<> type* sg_generic_malloc<type >(size_t len);								\
template<> type* sg_generic_calloc<type >(size_t len);								\
template<> type* sg_generic_realloc<type >(type* ptr, size_t old_len, size_t len);	\
template<> void sg_generic_free<type >(type* ptr);
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

void* get_copy(void* src, size_t len);
char* get_strdup(const char* str);
}

#endif // DOXYGEN_SHOULD_SKIP_THIS

#endif // __MEMORY_H__
