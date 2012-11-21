/*
  Copyright (c) 2009 Yahoo! Inc.  All rights reserved.  The copyrights
  embodied in the content of this file are licensed under the BSD
  (revised) open source license.

  Copyright (c) 2011 Berlin Institute of Technology and Max-Planck-Society.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  Shogun adjustments (w) 2011 Shashwat Lal Das
*/

#include <stdlib.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>

#ifndef VARRAY_H__
#define VARRAY_H__

namespace shogun
{
/** @brief Class v_array is a templated class used to
 * store variable length arrays. Memory locations
 * are stored as 'extents', i.e., address of the
 * first memory location and address after the last member.
 *
 * `begin', `end' and `end_array' handle the extent of the array.
 * `end_array' is the address at the end of the space allocated
 * for the array while `end' is the address at the end of the
 * space actually filled with elements.
 *
 * The class should be instantiated on the basis of the
 * data type of the contained array, eg. v_array<float32_t> if
 * one wants to store a float32_t* array.
 */
template<class T> class v_array
{
public:

	/**
	 * Constructor.
	 * Sets begin and end pointers to NULL.
	 */
	v_array()
	{
		begin = NULL;
		end = NULL;
		end_array = NULL;
	}

	/** Destructor
	 *
	 * will only free the array not the ptrs it contains
	 */
	~v_array()
	{
		SG_FREE(begin);
	}

	/**
	 * Operator [] overloaded to return the
	 * i-th element of the stored array.
	 *
	 * @param i index of the element, starting at 0
	 *
	 * @return element at position i
	 */
	T& operator[](unsigned int i) { return begin[i]; }

	/**
	 * Return the last element.
	 *
	 * @return Last element of array
	 */
	inline T last() { return *(end-1); }

	/**
	 * Pop from the array.
	 *
	 * @return Popped element
	 */
	inline T pop() { return *(--end); }

	/**
	 * Check if array is empty or not.
	 *
	 * @return whether array is empty
	 */
	inline bool empty() { return begin == end; }

	/**
	 * Decrement the 'end' pointer.
	 *
	 */
	inline void decr() { end--; }

	/**
	 * Get number of elements in array.
	 *
	 * @return number of array elements
	 */
	inline unsigned int index() { return end-begin; }

	/**
	 * Empty the array.
	 *
	 */
	inline void erase() { end = begin; }

	/**
	 * Push an element into the array.
	 *
	 * @param new_elem element to be pushed
	 */
	void push(const T &new_elem);

	/**
	 * Push multiple elements into the array.
	 *
	 * @param new_elem pointer to first element
	 * @param num number of elements
	 */
	void push_many(const T* new_elem, size_t num);

	/**
	 * Reserve space for specified number of elements.
	 * Reallocate, keeping the elements currently in the array.
	 *
	 * @param length number of elements to accommodate
	 */
	void reserve(size_t length);

	/**
	 * Reserve space for specified number of elements.
	 * No reallocation is done, array is replaced.
	 *
	 * @param length
	 */
	void calloc_reserve(size_t length);

	/**
	 * Pop an array from a list of arrays.
	 *
	 * @param stack array of arrays
	 * @return popped array
	 */
	v_array<T> pop(v_array< v_array<T> > &stack);

public:

	/// Pointer to first element of the array
	T* begin;

	/// Pointer to last set element in the array
	T* end;

	/// Pointer to end of array, based on memory reserved
	T* end_array;

};

template<class T>
inline void v_array<T>::push(const T &new_elem)
{
	if(end == end_array)
	{
		size_t old_length = end_array - begin;
		size_t new_length = 2 * old_length + 3;
		//size_t new_length = old_length + 1;
		begin = SG_REALLOC(T, begin, old_length, new_length);
		end = begin + old_length;
		end_array = begin + new_length;
	}
	*(end++) = new_elem;
}

template<class T>
inline void v_array<T>::push_many(const T* new_elem, size_t num)
{
	if(end+num >= end_array)
	{
		size_t length = end - begin;
		size_t new_length = CMath::max(2 * (size_t)(end_array - begin) + 3,
					       end - begin + num);
		begin = SG_REALLOC(T, begin, length, new_length);
		end = begin + length;
		end_array = begin + new_length;
	}
	memcpy(end, new_elem, num * sizeof(T));
	end += num;
}

template<class T>
inline void v_array<T>::reserve(size_t length)
{
	size_t old_length = end_array-begin;
	begin = SG_REALLOC(T, begin, old_length, length);
	if (old_length < length)
		memset(begin + old_length, 0, (length - old_length)*sizeof(T));

	end = begin;
	end_array = begin + length;
}

template<class T>
inline void v_array<T>::calloc_reserve(size_t length)
{
	begin = SG_CALLOC(T, length);
	end = begin;
	end_array = begin + length;
}

template<class T>
inline v_array<T> v_array<T>::pop(v_array< v_array<T> > &stack)
{
	if (stack.end != stack.begin)
		return *(--stack.end);
	else
		return v_array<T>();
}
}
#endif  // VARRAY_H__
