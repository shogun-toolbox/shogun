/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Soumyajit De, Evgeniy Andreev, Sergey Lisitsyn,
 *          Evan Shelhamer, Weijie Lin, Fernando Iglesias, Bjoern Esser,
 *          Thoralf Klein
 */

#ifndef __MEMORY_H__
#define __MEMORY_H__

#include <type_traits>

#include <shogun/lib/config.h>
#include <shogun/base/macros.h>
#include <shogun/lib/common.h>

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <new>
#include <cstring>

/* memcpy wrapper to enable clean moves to different memcpy backends */
namespace shogun
{

template <class InputIt, class OutputIt>
SG_FORCED_INLINE void* sg_memcpy(InputIt dest, OutputIt src, size_t count)
{
	return std::memcpy(static_cast<void*>(dest), static_cast<const void*>(src), count);
}

}  // namespace shogun

/* wrappers for malloc, free, realloc, calloc */

/* overload new() / delete */
void* operator new(size_t size);
void operator delete(void *p) noexcept;

/* overload new[] / delete[] */
void* operator new[](size_t size);
void operator delete[](void *p) noexcept;

#ifdef HAVE_ALIGNED_MALLOC
#ifdef HAVE_ALIGNED_NEW
void* operator new(size_t count, std::align_val_t al);
void* operator new[](size_t count, std::align_val_t al);
void operator delete(void *p, std::align_val_t al);
void operator delete[](void *p, std::align_val_t al);
#endif // HAVE_ALIGNED_NEW
#define SG_ALIGNED_MALLOC(type, len, al) sg_aligned_malloc<type>(size_t(len), al)
#else
#ifdef _MSC_VER
#pragma message("Aligned malloc is not available")
#elif defined(__GNUC__) || defined(__clang__)
#warning "Aligned malloc is not available"
#endif
#define SG_ALIGNED_MALLOC(type, len, al) sg_generic_malloc<type>(size_t(len))
#endif // HAVE_ALIGNED_MALLOC

#define SG_MALLOC(type, len) sg_generic_malloc<type>(size_t(len))
#define SG_CALLOC(type, len) sg_generic_calloc<type>(size_t(len))
#define SG_REALLOC(type, ptr, old_len, len) sg_generic_realloc<type>(ptr, size_t(old_len), size_t(len))
#define SG_FREE(ptr) sg_generic_free(ptr)

namespace shogun
{
	class SGReferencedData;
	template<class T>
	using is_sg_referenced = typename std::is_base_of<SGReferencedData, T>;

SHOGUN_EXPORT void* sg_malloc(size_t size);
template <class T, std::enable_if_t<!is_sg_referenced<T>::value, T>* = nullptr>
T* sg_generic_malloc(size_t len)
{
	return (T*) sg_malloc(sizeof(T)*len);
}

template<class T, std::enable_if_t<is_sg_referenced<T>::value, T>* = nullptr>
T* sg_generic_malloc(size_t len)
{
	return new T[len]();
}

SHOGUN_EXPORT void* sg_realloc(void* ptr, size_t size);
template<class T, std::enable_if_t<!is_sg_referenced<T>::value, T>* = nullptr>
T* sg_generic_realloc(T* ptr, size_t old_len, size_t len)
{
	return (T*) sg_realloc(ptr, sizeof(T)*len);
}

template<class T, std::enable_if_t<is_sg_referenced<T>::value, T>* = nullptr>
T* sg_generic_realloc(T* ptr, size_t old_len, size_t len)
{
	T* new_ptr = new T[len]();
	size_t min_len=old_len;
	if (len<min_len)
		min_len=len;
	for (size_t i=0; i<min_len; i++)
		new_ptr[i]=ptr[i];
	delete[] ptr;
	return new_ptr;
}

SHOGUN_EXPORT void* sg_calloc(size_t num, size_t size);
template<class T, std::enable_if_t<!is_sg_referenced<T>::value, T>* = nullptr>
T* sg_generic_calloc(size_t len)
{
	return (T*) sg_calloc(len, sizeof(T));
}

template<class T, std::enable_if_t<is_sg_referenced<T>::value, T>* = nullptr>
T* sg_generic_calloc(size_t len)
{
	return new T[len]();
}

SHOGUN_EXPORT void sg_free(void* ptr);
template<class T, std::enable_if_t<!is_sg_referenced<T>::value, T>* = nullptr>
void sg_generic_free(T* ptr)
{
	sg_free(ptr);
}

template<class T, std::enable_if_t<is_sg_referenced<T>::value, T>* = nullptr>
void sg_generic_free(T* ptr)
{
	delete[] ptr;
}

#ifdef HAVE_ALIGNED_MALLOC
SHOGUN_EXPORT void* sg_aligned_malloc(size_t size, size_t al);
template <class T, std::enable_if_t<!is_sg_referenced<T>::value, T>* = nullptr>
T* sg_aligned_malloc(size_t len, size_t al)
{
	return (T*) sg_aligned_malloc(sizeof(T)*len, al);
}
#endif // HAVE_ALIGNED_MALLOC

namespace alignment
{
	static constexpr index_t container_alignment = 16;
}

void* get_copy(void* src, size_t len);
char* get_strdup(const char* str);
}

#endif // DOXYGEN_SHOULD_SKIP_THIS

#endif // __MEMORY_H__
