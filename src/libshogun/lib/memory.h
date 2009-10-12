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
#include <stdlib.h>

#include <new>

void* operator new(size_t size) throw (std::bad_alloc);
void operator delete(void *p) throw();

void* operator new[](size_t size) throw (std::bad_alloc);
void operator delete[](void *p) throw();

#ifndef __MEMORY_H__
#define __MEMORY_H__

#ifdef TRACE_MEMORY_ALLOCS
class CMemoryBlock
{
	public:
		CMemoryBlock(void* p)
		{
			ptr=p;
			size=0;
		}

		CMemoryBlock(void* p, size_t sz)
		{
			ptr=p;
			size=sz;
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
			printf("Object at %p of size %lld bytes\n", ptr, (long long int) size);
		}

	protected:
		void* ptr;
		size_t size;
};

void list_memory_allocs();
#endif
#endif // __MEMORY_H__
