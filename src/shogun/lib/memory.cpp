/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evgeniy Andreev, Viktor Gal, Soumyajit De,
 *          Weijie Lin, Bjoern Esser, Sergey Lisitsyn, Thoralf Klein
 */

#include <shogun/lib/config.h>
#include <shogun/lib/ShogunException.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGMatrix.h>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_JEMALLOC
#include <jemalloc/jemalloc.h>
#elif USE_TCMALLOC
#include <gperftools/tcmalloc.h>
#endif

using namespace shogun;

#ifdef TRACE_MEMORY_ALLOCS
#include <shogun/lib/Map.h>
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

SG_FORCED_INLINE bool allocation_error(void *p, size_t size, const char* op_str)
{
	const size_t buf_len=128;
	char buf[buf_len];
	size_t written=snprintf(buf, buf_len,
		"Out of memory error, tried to allocate %lld bytes using %s.", (long long int) size, op_str);
	if (written<buf_len)
		throw ShogunException(buf);
	else
		throw ShogunException("Out of memory error");
}

#ifndef USE_JEMALLOC
void* operator new(size_t size)
{
#if defined(USE_TCMALLOC)
	void *p=tc_malloc(size);
#else
	void *p=std::malloc(size);
#endif

#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
		sg_mallocs->add(p, MemoryBlock(p,size));
#endif
	if (!p)
		allocation_error(p, size, "new()");
	return p;
}

void operator delete(void *p) throw()
{
#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
		sg_mallocs->remove(p);
#endif

#if defined(USE_TCMALLOC)
	tc_free(p);
#else
	std::free(p);
#endif
}

void* operator new[](size_t size)
{
	return ::operator new(size);
}

void operator delete[](void *p) throw()
{
	::operator delete(p);
}

#ifdef HAVE_STD_ALIGNED_ALLOC
void* operator new(size_t size, std::align_val_t al)
{
	std::size_t align = (std::size_t)al;
	/* C11: the value of size shall be an integral multiple of alignment.  */
	if (std::size_t rem = size & (align - 1))
		size += align - rem;
#if defined(USE_TCMALLOC)
	void *p = tc_new_aligned_nothrow(size, align);
#else
	void *p = std::aligned_alloc(size, align);
#endif

#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
		sg_mallocs->add(p, MemoryBlock(p,size));
#endif
	if (!p)
		allocation_error(p, size, "new");

	return p;
}

void* operator new[](size_t size, std::align_val_t al)
{
	return ::operator new(size, al);
}

void operator delete(void *p, std::align_val_t al)
{
	::operator delete(p);
}

void operator delete[](void *p, std::align_val_t al)
{
	::operator delete(p);
}
#endif // HAVE_STD_ALIGNED_ALLOC

#endif // USE_JEMALLOC

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
	void* p=std::malloc(size);
#endif
#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
		sg_mallocs->add(p, MemoryBlock(p,size, file, line));
#endif
	if (!p)
		allocation_error(p, size, "malloc");

	return p;
}

#ifdef HAVE_ALIGNED_MALLOC
void* sg_aligned_malloc(size_t size, size_t al)
{
	/* the value of size shall be an integral multiple of alignment.  */
	if (std::size_t rem = size & (al - 1))
		size += al - rem;
#if defined(USE_JEMALLOC)
	void* p = je_aligned_alloc(al, size);
#elif defined(USE_TCMALLOC)
	void *p = tc_memalign(al, size);
#else

#ifdef HAVE_STD_ALIGNED_ALLOC
	void* p = std::aligned_alloc(al, size);
#else

#ifdef _MSC_VER
	void* p = _aligned_malloc(size, al);
#elif defined(HAVE_POSIX_MEMALIGN)
	void* p = nullptr;
	int r = posix_memalign(&p, al, size);
	if (r)
		p = nullptr;
#endif
#endif // HAVE_STD_ALIGNED_ALLOC
#endif // USE_JEMALLOC || USE_TCMALLOC

#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
		sg_mallocs->add(p, MemoryBlock(p,size, file, line));
#endif
	allocation_error(p, size, "aligned_malloc");

	return p;
}
#endif // HAVE_ALIGNED_MALLOC

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
		allocation_error(p, size, "calloc");
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
		allocation_error(p, size, "realloc");

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

}

void* shogun::get_copy(void* src, size_t len)
{
	void* copy=SG_MALLOC(uint8_t, len);
	sg_memcpy(copy, src, len);
	return copy;
}

char* shogun::get_strdup(const char* str)
{
	if (!str)
		return NULL;

	return (char*) get_copy((void*) str, strlen(str)+1);
}
