/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DYNAMIC_ARRAY_H_
#define _DYNAMIC_ARRAY_H_

#include "base/SGObject.h"
#include "base/DynArray.h"
#include "base/Parameter.h"

namespace shogun
{
/** @brief Template Dynamic array class that creates an array that can
 * be used like a list or an array.
 *
 * It grows and shrinks dynamically, while elements can be accessed
 * via index.  It is performance tuned for simple types like float
 * etc. and for hi-level objects only stores pointers, which are not
 * automagically SG_REF'd/deleted.
 */
template <class T> class CDynamicArray :public CSGObject
{
	public:
		/** constructor
		 *
		 * @param p_resize_granularity resize granularity
		 */
		CDynamicArray(int32_t p_resize_granularity=128)
		: CSGObject()
		{
			set_generic<T>();

			m_parameters->add_vector(&m_array.array,
									 &m_array.num_elements, "array",
									 "Memory for dynamic array.");
			m_parameters->add(&m_array.last_element_idx,
							  "last_element_idx",
							  "Element with largest index.");
			m_parameters->add(&m_array.resize_granularity,
							  "resize_granularity",
							  "shrink/grow step size.");
		}

		virtual ~CDynamicArray() {}

		/** set the resize granularity
		 *
		 * @param g new granularity
		 * @return what has been set (minimum is 128)
		 */
		inline int32_t set_granularity(int32_t g)
		{
			return m_array.set_granularity(g);
		}

		/** get array size (including granularity buffer)
		 *
		 * @return total array size (including granularity buffer)
		 */
		inline int32_t get_array_size()
		{
			return m_array.get_array_size();
		}

		/** get number of elements
		 *
		 * @return number of elements
		 */
		inline int32_t get_num_elements() const
		{
			return m_array.get_num_elements();
		}

		/** get array element at index
		 *
		 * (does NOT do bounds checking)
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline T get_element(int32_t index) const
		{
			return m_array.get_element(index);
		}

		/** get array element at index
		 *
		 * (does bounds checking)
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline T get_element_safe(int32_t index) const
		{
			return m_array.get_element_safe(index);
		}

		/** set array element at index
		 *
		 * @param element element to set
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool set_element(T element, int32_t index)
		{
			return m_array.set_element(element, index);
		}

		/** insert array element at index
		 *
		 * @param element element to insert
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool insert_element(T element, int32_t index)
		{
			return m_array.insert_element(element, index);
		}

		/** append array element to the end of array
		 *
		 * @param element element to append
		 * @return if setting was successful
		 */
		inline bool append_element(T element)
		{
			return m_array.append_element(element);
		}

	    /** ::STD::VECTOR compatible. Append array element to the end
		 *  of array.
		 *
		 * @param element element to append
		 */
		inline void push_back(T element)
		{ m_array.push_back(element); }

	    /** ::STD::VECTOR compatible. Delete array element at the end
		 *  of array.
		 */
		inline void pop_back()
		{
			m_array.pop_back();
		}

		/** ::STD::VECTOR compatible. Return array element at the end
		 *  of array.
		 *
		 * @return element at the end of array
		 */
		inline T back()
		{
			return m_array.back();
		}

		/** find first occurence of array element and return its index
		 * or -1 if not available
		 *
		 * @param element element to search for
		 * @return index of element or -1
		 */
		inline int32_t find_element(T element)
		{
			return m_array.find_element(element);
		}

		/** delete array element at idx
		 * (does not call delete[] or the like)
		 *
		 * @param idx index
		 * @return if deleting was successful
		 */
		inline bool delete_element(int32_t idx)
		{
			return m_array.delete_element(idx);
		}

		/** resize the array
		 *
		 * @param n new size
		 * @return if resizing was successful
		 */
		inline bool resize_array(int32_t n)
		{
			return m_array.resize_array(n);
		}

		/** get the array
		 * call get_array just before messing with it DO NOT call any
		 * [],resize/delete functions after get_array(), the pointer may
		 * become invalid !
		 *
		 * @return the array
		 */
		inline T* get_array()
		{
			return m_array.get_array();
		}

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array new array
		 * @param p_num_elements last element index + 1
		 * @param array_size number of elements in array
		 */
		inline void set_array(T* p_array, int32_t p_num_elements,
							  int32_t array_size)
		{
			m_array.set_array(p_array, p_num_elements, array_size);
		}

		/** clear the array (with zeros) */
		inline void clear_array()
		{
			m_array.clear_array();
		}

		/** operator overload for array read only access
		 * use set_element() for write access (will also make the array
		 * dynamically grow)
		 *
		 * DOES NOT DO ANY BOUNDS CHECKING
		 *
		 * @param index index
		 * @return element at index
		 */
		inline T operator[](int32_t index) const
		{
			return m_array[index];
		}

		/** operator overload for array assignment
		 *
		 * @param orig original array
		 * @return new array
		 */
		inline CDynamicArray<T>& operator=(CDynamicArray<T>& orig)
		{
			m_array = orig.m_array;
			return *this;
		}

		/** @return object name */
		inline virtual const char* get_name() const
		{
			return "DynamicArray";
		}

	protected:
		DynArray<T> m_array;
};
}
#endif /* _DYNAMIC_ARRAY_H_  */
