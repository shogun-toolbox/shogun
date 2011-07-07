/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DYNAMIC_OBJECT_ARRAY_H_
#define _DYNAMIC_OBJECT_ARRAY_H_

#include "base/SGObject.h"
#include "base/DynArray.h"
#include "base/Parameter.h"

namespace shogun
{
/** @brief Template Dynamic array class that creates an array that can
 * be used like a list or an array.
 *
 * It grows and shrinks dynamically, while elements can be accessed
 * via index.  It only stores CSGObject pointers, which ARE automagically
 * SG_REF'd/deleted.
 *
 * Note that this array is generic, but only takes pointers to objects which
 * implement the CSGObject interface, so only put these in here.
 */
template<class T>class CDynamicObjectArray :public CSGObject
{
	DynArray<T> m_array;

	public:
		/** constructor
		 *
		 * @param p_resize_granularity resize granularity
		 */
		CDynamicObjectArray(int32_t p_resize_granularity=128)
		: CSGObject()
		{
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

		virtual ~CDynamicObjectArray() {}

		/** set the resize granularity
		 *
		 * @param g new granularity
		 * @return what has been set (minimum is 128)
		 */
		inline int32_t set_granularity(int32_t g)
		{ return m_array.set_granularity(g); }

		/** get array size (including granularity buffer)
		 *
		 * @return total array size (including granularity buffer)
		 */
		inline int32_t get_array_size(void)
		{ return m_array.get_array_size(); }

		/** get number of elements
		 *
		 * @return number of elements
		 */
		inline int32_t get_num_elements(void) const
		{ return m_array.get_num_elements(); }

		/** get array element at index
		 *
		 * (does NOT do bounds checking)
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline T get_element(int32_t index) const
		{
			T element=m_array.get_element(index);
			SG_REF(element);
			return element;
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
			T element=m_array.get_element_safe(index);
			SG_REF(element);
			return element;
		}

		/** set array element at index
		 *
		 * @param element element to set
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool set_element(T element, int32_t index)
		{
			T old=m_array.get_element(index);
			SG_UNREF(old);

			bool success=m_array.set_element(element, index);
			if (success)
				SG_REF(element);

			return success;
		}

		/** insert array element at index
		 *
		 * @param element element to insert
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool insert_element(T element, int32_t index)
		{
			bool success=m_array.insert_element(element, index);
			if (success)
				SG_REF(element);

			return success;
		}

		/** append array element to the end of array
		 *
		 * @param element element to append
		 * @return if setting was successful
		 */
		inline bool append_element(T element)
		{
			bool success=m_array.append_element(element);
			if (success)
				SG_REF(element);

			return success;
		}

	    /** ::STD::VECTOR compatible. Append array element to the end
		 *  of array.
		 *
		 * @param element element to append
		 */
		inline void push_back(T element)
		{
			SG_REF(element);
			m_array.push_back(element);
		}

	    /** ::STD::VECTOR compatible. Delete array element at the end
		 *  of array.
		 */
		inline void pop_back(void)
		{
			T element=m_array.back();
			SG_UNREF(element);

			m_array.pop_back();
		}

		/** ::STD::VECTOR compatible. Return array element at the end
		 *  of array.
		 *
		 * @return element at the end of array
		 */
		inline T back(void)
		{
			T element=m_array.back();
			SG_REF(element);
			return element;
		}

		/** find first occurence of array element and return its index
		 * or -1 if not available
		 *
		 * @param element element to search for
		 * @return index of element or -1
		 */
		inline int32_t find_element(T element)
		{ return m_array.find_element(element); }

		/** delete array element at idx
		 * (does not call delete[] or the like)
		 *
		 * @param idx index
		 * @return if deleting was successful
		 */
		inline bool delete_element(int32_t idx)
		{
			T element=m_array.get_element(idx);
			SG_UNREF(element);

			return m_array.delete_element(idx);
		}

		/** resize the array
		 *
		 * @param n new size
		 * @return if resizing was successful
		 */
		inline bool resize_array(int32_t n)
		{ return m_array.resize_array(n); }

		/** get the array
		 * call get_array just before messing with it DO NOT call any
		 * [],resize/delete functions after get_array(), the pointer may
		 * become invalid !
		 * Also, be carefull with SG_REF/SG_UNREF stuff if working directly on
		 * array!
		 *
		 * @return the array
		 */
		inline T* get_array(void)
		{ return m_array.get_array(); }

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array new array
		 * @param p_num_elements last element index + 1
		 * @param array_size number of elements in array
		 */
		inline void set_array(
			T* p_array, int32_t p_num_elements,
			int32_t array_size)
		{
			unref_all();
			m_array.set_array(p_array, p_num_elements, array_size);
		}

		/** clear the array (with zeros) */
		inline void clear_array(void)
		{
			unref_all();
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
			T element=m_array[index];
			SG_REF(element);

			return element;
		}

		/** operator overload for array assignment
		 *
		 * @param orig original array
		 * @return new array
		 */
		inline CDynamicObjectArray& operator=(CDynamicObjectArray& orig)
		{
			unref_all();

			/* SG_REF all new elements */
			for (index_t i=0; i<orig.get_num_elements(); ++i)
			{
				T element=orig.get_element(i);
				SG_REF(element);
			}

			m_array = orig.m_array;
			return *this;
		}

		/** @return object name */
		inline virtual const char* get_name() const
		{ return "DynamicObjectArray"; }

	private:
		inline void unref_all()
		{
			/* SG_REF all new elements */
			for (index_t i=0; i<m_array.get_num_elements(); ++i)
			{
				T element=m_array.get_element(i);
				SG_UNREF(element);
			}
		}
};
}
#endif /* _DYNAMIC_OBJECT_ARRAY_H_  */
