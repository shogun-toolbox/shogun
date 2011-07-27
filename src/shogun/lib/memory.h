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

#include <stdio.h>
#include <stdlib.h>

#include <new>

/* wrappers for malloc, free, realloc, calloc */
#ifdef TRACE_MEMORY_ALLOCS
#define SG_MALLOC(type, len) (type*) sg_malloc(sizeof(type)*size_t(len), __FILE__, __LINE__)
#define SG_MALLOC(type, len) (type*) sg_malloc(sizeof(type)*size_t(len), __FILE__, __LINE__)
#define SG_CALLOC(type, len) (type*) sg_calloc(size_t(len), sizeof(type), __FILE__, __LINE__)
#define SG_REALLOC(type, ptr, len) (type*) sg_realloc(ptr, sizeof(type)*size_t(len), __FILE__, __LINE__)
#define SG_FREE(ptr) sg_free(ptr)

void* sg_malloc(size_t size, const char* file, int line);
void  sg_free(void* ptr);
void* sg_realloc(void* ptr, size_t size, const char* file, int line);
void* sg_calloc(size_t num, size_t size, const char* file, int line);
#else //TRACE_MEMORY_ALLOCS

#define SG_MALLOC(type, len) (type*) sg_malloc(sizeof(type)*size_t(len))
#define SG_MALLOC(type, len) (type*) sg_malloc(sizeof(type)*size_t(len))
#define SG_CALLOC(type, len) (type*) sg_calloc(size_t(len), sizeof(type))
#define SG_REALLOC(type, ptr, len) (type*) sg_realloc(ptr, sizeof(type)*size_t(len))
#define SG_FREE(ptr) sg_free(ptr)

void* sg_malloc(size_t size);
void  sg_free(void* ptr);
void* sg_realloc(void* ptr, size_t size);
void* sg_calloc(size_t num, size_t size);
#endif //TRACE_MEMORY_ALLOCS

/* overload new() / delete */
void* operator new(size_t size) throw (std::bad_alloc);
void operator delete(void *p);

/* overload new[] / delete[] */
void* operator new[](size_t size);
void operator delete[](void *p);


#ifdef TRACE_MEMORY_ALLOCS
namespace shogun
{
class MemoryBlock
{
	public:
		MemoryBlock(void* p);
		MemoryBlock(void* p, size_t sz, const char* fname=NULL, int linenr=-1);
        MemoryBlock(const MemoryBlock &b);
		
		bool operator==(const MemoryBlock &b) const;
		void display();
		void set_sgobject();

	protected:
		void* ptr;
		size_t size;
		const char* file;
		int line;
		bool is_sgobject;
};
}

void list_memory_allocs();
#endif
#endif // __MEMORY_H__
