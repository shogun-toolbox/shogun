/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DYNAMIC_OBJECT_ARRAY_H_
#define _DYNAMIC_OBJECT_ARRAY_H_

#include <shogun/base/SGObject.h>
#include <shogun/base/DynArray.h>
#include <shogun/base/Parameter.h>

namespace shogun
{
/** @brief Dynamic array class for CSGObject pointers that creates an array
 * that can be used like a list or an array.
 *
 * It grows and shrinks dynamically, while elements can be accessed
 * via index. It only stores CSGObject pointers, which ARE automagically
 * SG_REF'd/deleted.
 *
 */
class CDynamicObjectArray : public CSGObject
{
	public:
		/** default constructor
		 *
		 * @param p_resize_granularity resize granularity
		 */
		CDynamicObjectArray()
		: CSGObject(), m_array()
		{
			m_parameters->add_vector(&m_array.array, &m_array.num_elements,
					"array", "Memory for dynamic array.");
			m_parameters->add(&m_array.last_element_idx, "last_element_idx",
					"Element with largest index.");
			m_parameters->add(&m_array.resize_granularity, "resize_granularity",
					"shrink/grow step size.");

			dim1_size=1;
			dim2_size=1;
			dim3_size=1;
			name="Array";
		}
		/** constructor
		 *
		 * @param p_resize_granularity resize granularity
		 */
		CDynamicObjectArray(int32_t dim1, int32_t dim2=1, int32_t dim3=1)
		: CSGObject(), m_array(dim1*dim2*dim3)
		{
			m_parameters->add_vector(&m_array.array, &m_array.num_elements,
					"array", "Memory for dynamic array.");
			m_parameters->add(&m_array.last_element_idx, "last_element_idx",
					"Element with largest index.");
			m_parameters->add(&m_array.resize_granularity, "resize_granularity",
					"shrink/grow step size.");

			dim1_size=dim1;
			dim2_size=dim2;
			dim3_size=dim3;
			name="Array";
		}

		/** 1d */
		CDynamicObjectArray(CSGObject** p_array, int32_t p_dim1_size, bool p_free_array=true, bool p_copy_array=false)
		: CSGObject(), m_array(p_array, p_dim1_size, p_free_array, p_copy_array)
		{
			dim1_size=p_dim1_size;
			dim2_size=1;
			dim3_size=1;
			name="Array";
		}

		/** 2d */
		CDynamicObjectArray(CSGObject** p_array, int32_t p_dim1_size, int32_t p_dim2_size,
						bool p_free_array=true, bool p_copy_array=false)
		: CSGObject(), m_array(p_array, p_dim1_size*p_dim2_size, p_free_array, p_copy_array)
		{
			dim1_size=p_dim1_size;
			dim2_size=p_dim2_size;
			dim3_size=1;
			name="Array";
		}

		/** 3d */
		CDynamicObjectArray(CSGObject** p_array, int32_t p_dim1_size, int32_t p_dim2_size,
						int32_t p_dim3_size, bool p_free_array=true, bool p_copy_array=false)
		: CSGObject(), m_array(p_array, p_dim1_size*p_dim2_size*p_dim3_size, p_free_array, p_copy_array)
		{
			dim1_size=p_dim1_size;
			dim2_size=p_dim2_size;
			dim3_size=p_dim3_size;
			name="Array";
		}

		virtual ~CDynamicObjectArray() { unref_all(); }

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
		inline int32_t get_array_size()
		{
			return m_array.get_array_size();
		}

		/** get array size (including granularity buffer)
		 *
		 * @return total array size (including granularity buffer)
		 */
		inline void get_array_size(int32_t& dim1, int32_t& dim2)
		{
			dim1=dim1_size;
			dim2=dim2_size;
		}

		/** get array size (including granularity buffer)
		 *
		 * @return total array size (including granularity buffer)
		 */
		inline void get_array_size(int32_t& dim1, int32_t& dim2, int32_t& dim3)
		{
			dim1=dim1_size;
			dim2=dim2_size;
			dim3=dim3_size;
		}

		/** */
		inline int32_t get_dim1() { return dim1_size; }

		/** */
		inline int32_t get_dim2() { return dim2_size; }

		/** */
		inline int32_t get_dim3() { return dim3_size; }

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
		inline CSGObject* get_element(int32_t index) const
		{
			CSGObject* elem=m_array.get_element(index);
			SG_REF(elem);
			return elem;
		}

		/** */
		inline CSGObject* element(int32_t index1, int32_t index2=0, int32_t index3=0)
		{
			return get_element(index1+dim1_size*(index2+dim2_size*index3));
		}

		/** get last array element
		 *
		 * @return last array element
		 */
		inline CSGObject* get_last_element() const
		{
			CSGObject* elem=m_array.get_last_element();
			SG_REF(elem);
			return elem;
		}

		/** get array element at index
		 *
		 * (does bounds checking)
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline CSGObject* get_element_safe(int32_t index) const
		{
			CSGObject* elem=m_array.get_element_safe(index);
			SG_REF(elem);
			return elem;
		}

		/** set array element at index
		 *
		 * @param element element to set
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool set_element(CSGObject* elem, int32_t index1, int32_t index2=0, int32_t index3=0)
		{
			CSGObject* old=(CSGObject*)m_array.get_element(index1+dim1_size*(index2+dim2_size*index3));

			bool success=m_array.set_element(elem, index1+dim1_size*(index2+dim2_size*index3));
			if (success)
			{
				SG_REF(elem);
				SG_UNREF(old);
			}

			/* ref before unref to prevent deletion if new=old */
			return success;
		}

		/** insert array element at index
		 *
		 * @param element element to insert
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool insert_element(CSGObject* elem, int32_t index)
		{
			bool success=m_array.insert_element(elem, index);
			if (success)
				SG_REF(elem);

			return success;
		}

		/** append array element to the end of array
		 *
		 * @param element element to append
		 * @return if setting was successful
		 */
		inline bool append_element(CSGObject* elem)
		{
			bool success=m_array.append_element(elem);
			if (success)
				SG_REF(elem);

			return success;
		}

		/** STD VECTOR compatible. Append array element to the end
		 *  of array.
		 *
		 * @param element element to append
		 */
		inline void push_back(CSGObject* elem)
		{
			SG_REF(elem);
			m_array.push_back(elem);
		}

		/** STD VECTOR compatible. Delete array element at the end
		 *  of array.
		 */
		inline void pop_back()
		{
			CSGObject* elem=m_array.back();
			SG_UNREF(elem);

			m_array.pop_back();
		}

		/** STD VECTOR compatible. Return array element at the end
		 *  of array.
		 *
		 * @return element at the end of array
		 */
		inline CSGObject* back() const
		{
			CSGObject* elem=m_array.back();
			SG_REF(elem);
			return elem;
		}

		/** find first occurence of array element and return its index
		 * or -1 if not available
		 *
		 * @param element element to search for
		 * @return index of element or -1
		 */
		inline int32_t find_element(CSGObject* elem) const
		{
			return m_array.find_element(elem);
		}

		/** delete array element at idx
		 * (does not call SG_FREE() or the like)
		 *
		 * @param idx index
		 * @return if deleting was successful
		 */
		inline bool delete_element(int32_t idx)
		{
			CSGObject* elem=m_array.get_element(idx);
			SG_UNREF(elem);

			return m_array.delete_element(idx);
		}

		inline bool resize_array(int32_t ndim1, int32_t ndim2=1, int32_t ndim3=1)
		{
			dim1_size=ndim1;
			dim2_size=ndim2;
			dim3_size=ndim3;
			return m_array.resize_array(ndim1*ndim2*ndim3);
		}

		/** clear the array (with zeros) */
		inline void clear_array()
		{
			unref_all();
			m_array.clear_array();
		}

		/** resets the array */
		inline void reset_array()
		{
			unref_all();
			m_array.reset();
		}

		/** operator overload for array assignment
		 *
		 * @param orig original array
		 * @return new array
		 */
		inline CDynamicObjectArray& operator=(CDynamicObjectArray& orig)
		{
			/* SG_REF all new elements (implicitly) */
			for (index_t i=0; i<orig.get_num_elements(); ++i)
				orig.get_element(i);

			/* unref after adding to avoid possible deletion */
			unref_all();

			/* copy pointer DynArray */
			m_array=orig.m_array;
			return *this;
		}

		/** @return underlying array of pointers */
		inline CSGObject** get_array() const { return m_array.get_array(); }

		/** shuffles the array */
		inline void shuffle() { m_array.shuffle(); }

		inline void set_array_name(const char* p_name)
		{
			name=p_name;
		}

		inline const char* get_array_name() const { return name; }

		/** @return object name */
		inline virtual const char* get_name() const
		{ return "DynamicObjectArray"; }

	private:
		/** de-reference all elements of this array once */
		inline void unref_all()
		{
			/* SG_UNREF all my elements */
			for (index_t i=0; i<m_array.get_num_elements(); ++i)
			{
				CSGObject* elem=m_array.get_element(i);
				SG_UNREF(elem);
			}
		}

	private:
		DynArray<CSGObject*> m_array;

		/** */
		int32_t dim1_size;

		/** */
		int32_t dim2_size;

		/** */
		int32_t dim3_size;

		/** */
		const char* name;

};
}
#endif /* _DYNAMIC_OBJECT_ARRAY_H_  */
