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

#include <shogun/base/SGObject.h>
#include <shogun/base/DynArray.h>
#include <shogun/base/Parameter.h>

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
		/** default constructor
		 *
		 * @param p_resize_granularity resize granularity
		 */
		CDynamicArray()
		: CSGObject(), m_array()
		{
			//set_generic<T>();

			/*m_parameters->add_vector(&m_array.array),
									 &m_array.num_elements, "array",
									 "Memory for dynamic array.");
			m_parameters->add(&m_array.last_element_idx,
							  "last_element_idx",
							  "Element with largest index.");
			m_parameters->add(&m_array.resize_granularity,
							  "resize_granularity",
							  "shrink/grow step size.");*/

			dim1_size=1;
			dim2_size=1;
			dim3_size=1;
			name="Array";
		}
		/** constructor
		 *
		 * @param p_resize_granularity resize granularity
		 */
		CDynamicArray(int32_t dim1, int32_t dim2=1, int32_t dim3=1)
		: CSGObject(), m_array(dim1*dim2*dim3)
		{
			//set_generic<T>();

			/*m_parameters->add_vector(&m_array.array),
									 &m_array.num_elements, "array",
									 "Memory for dynamic array.");
			m_parameters->add(&m_array.last_element_idx,
							  "last_element_idx",
							  "Element with largest index.");
			m_parameters->add(&m_array.resize_granularity,
							  "resize_granularity",
							  "shrink/grow step size.");*/

			dim1_size=dim1;
			dim2_size=dim2;
			dim3_size=dim3;
		}

		/** 1d */
		CDynamicArray(T* p_array, int32_t p_dim1_size, bool p_free_array=true, bool p_copy_array=false)
		: CSGObject(), m_array(p_array, p_dim1_size, p_free_array, p_copy_array)
		{
			dim1_size=p_dim1_size;
			dim2_size=1;
			dim3_size=1;
		}

		/** 2d */
		CDynamicArray(T* p_array, int32_t p_dim1_size, int32_t p_dim2_size,
						bool p_free_array=true, bool p_copy_array=false)
		: CSGObject(), m_array(p_array, p_dim1_size*p_dim2_size, p_free_array, p_copy_array)
		{
			dim1_size=p_dim1_size;
			dim2_size=p_dim2_size;
			dim3_size=1;
		}

		/** 3d */
		CDynamicArray(T* p_array, int32_t p_dim1_size, int32_t p_dim2_size,
						int32_t p_dim3_size, bool p_free_array=true, bool p_copy_array=false)
		: CSGObject(), m_array(p_array, p_dim1_size*p_dim2_size*p_dim3_size, p_free_array, p_copy_array)
		{
			dim1_size=p_dim1_size;
			dim2_size=p_dim2_size;
			dim3_size=p_dim3_size;
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_array_size size of another array
		 */
		CDynamicArray(const T* p_array, int32_t p_dim1_size=1, int32_t p_dim2_size=1, int32_t p_dim3_size=1)
		: CSGObject(), m_array(p_array, p_dim1_size*p_dim2_size*p_dim3_size)
		{
			dim1_size=p_dim1_size;
			dim2_size=p_dim2_size;
			dim3_size=p_dim3_size;
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

		/** get array 3d element at index
		 *
		 * (does NOT do bounds checking)
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline const T& get_element(int32_t index1, int32_t index2=0, int32_t index3=0) const
		{
			return m_array.get_array()[index1+dim1_size*(index2+dim2_size*index3)];
		}

		/** */
		inline const T& element(int32_t index1, int32_t index2=0, int32_t index3=0) const
		{
			return get_element(index1, index2, index3);
		}

		/** */
		inline T& element(int32_t index1, int32_t index2=0, int32_t index3=0)
		{
			return m_array.get_array()[index1+dim1_size*(index2+dim2_size*index3)];
		}

		/** get element of given array at given index
		 *
		 * @param p_array another array
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 3
		 * @return array element at index
		 */
		inline T& element(T* p_array, int32_t index1, int32_t index2=0, int32_t index3=0)
		{
			ASSERT(index1>=0 && index1<dim1_size);
			ASSERT(index2>=0 && index2<dim2_size);
			ASSERT(index3>=0 && index3<dim3_size);
			return p_array[index1+dim1_size*(index2+dim2_size*index3)];
		}

		/** get element of given array at given index
		 *
		 * @param p_array another array
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 3
		 * @param p_dim1_size size of dimension 1
		 * @param p_dim2_size size of dimension 2
		 * @return element of given array at given index
		 */
		inline T& element(T* p_array, int32_t index1, int32_t index2, int32_t index3, int32_t p_dim1_size, int32_t p_dim2_size)
		{
			ASSERT(p_dim1_size==dim1_size);
			ASSERT(p_dim2_size==dim2_size);
			ASSERT(index1>=0 && index1<p_dim1_size);
			ASSERT(index2>=0 && index2<p_dim2_size);
			ASSERT(index3>=0 && index3<dim3_size);
			return p_array[index1+p_dim1_size*(index2+p_dim2_size*index3)];
		}	

		/** gets last array element
		 *
		 * @return array element at last index
		 */
		inline T get_last_element() const
		{
			return m_array.get_last_element();
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

		/** set array 3d element at index
		 *
		 * @param element element to set
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool set_element(T elem, int32_t index1, int32_t index2=0, int32_t index3=0)
		{
			return m_array.set_element(elem, index1+dim1_size*(index2+dim2_size*index3));
		}

		/** insert array element at index
		 *
		 * @param element element to insert
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool insert_element(T elem, int32_t index)
		{
			return m_array.insert_element(elem, index);
		}

		/** append array element to the end of array
		 *
		 * @param element element to append
		 * @return if setting was successful
		 */
		inline bool append_element(T elem)
		{
			return m_array.append_element(elem);
		}

		/** STD VECTOR compatible. Append array element to the end
		 *  of array.
		 *
		 * @param element element to append
		 */
		inline void push_back(T elem)
		{ m_array.push_back(elem); }

		/** STD VECTOR compatible. Delete array element at the end
		 *  of array.
		 */
		inline void pop_back()
		{
			m_array.pop_back();
		}

		/** STD  VECTOR compatible. Return array element at the end
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
		inline int32_t find_element(T elem)
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
			return m_array.delete_element(idx);
		}

		inline bool resize_array(int32_t ndim1, int32_t ndim2=1, int32_t ndim3=1)
		{
			dim1_size=ndim1;
			dim2_size=ndim2;
			dim3_size=ndim3;
			return m_array.resize_array(ndim1*ndim2*ndim3);
		}

		void set_const(const T& const_element)
		{
			m_array.set_const(const_element);
		}

		/** get the array
		 * call get_array just before messing with it DO NOT call any
		 * [],resize/delete functions after get_array(), the pointer may
		 * become invalid !
		 *
		 * @return the array
		 */
		inline T* get_array() const
		{
			return m_array.get_array();
		}

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array new array
		 * @param p_num_elements last element index + 1
		 * @param array_size number of elements in array
		 */
		inline void set_array(T* p_array, int32_t p_dim1,
						bool p_free_array=true, bool p_copy_array=false)
		{
			dim1_size=p_dim1;
			dim2_size=1;
			dim3_size=1;
			m_array.set_array(p_array, p_dim1, p_dim1, p_free_array, p_copy_array);
		}

		/** set the 2d array pointer and free previously allocated memory
		 *
		 * @param p_array new array
		 * @param p_num_elements last element index + 1
		 * @param array_size number of elements in array
		 */
		inline void set_array(T* p_array, int32_t p_dim1,
						int32_t p_dim2, bool p_free_array=true, bool p_copy_array=false)
		{
			dim1_size=p_dim1;
			dim2_size=p_dim2;
			dim3_size=1;
			m_array.set_array(p_array, p_dim1*p_dim2, p_dim1*p_dim2, p_free_array, p_copy_array);
		}

		/** set the 3d array pointer and free previously allocated memory
		 *
		 * @param p_array new array
		 * @param p_num_elements last element index + 1
		 * @param array_size number of elements in array
		 */
		inline void set_array(T* p_array, int32_t p_dim1,
						int32_t p_dim2, int32_t p_dim3, bool p_free_array=true, bool p_copy_array=false)
		{
			dim1_size=p_dim1;
			dim2_size=p_dim2;
			dim3_size=p_dim3;
			m_array.set_array(p_array, p_dim1*p_dim2*p_dim3, p_dim1*p_dim2*p_dim3, p_free_array, p_copy_array);
		}

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array another array
		 * @param p_array_size size of another array
		 */
		inline void set_array(const T* p_array, int32_t p_size)
		{
			m_array.set_array(p_array, p_size, p_size);
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
		inline const T& operator[](int32_t index) const
		{
			return get_element(index);
		}

		/** */
		inline T& operator[](int32_t index)
		{
			return element(index);
		}

		/** operator overload for array assignment
		 *
		 * @param orig original array
		 * @return new array
		 */
		inline CDynamicArray<T>& operator=(CDynamicArray<T>& orig)
		{
			m_array=orig.m_array;
			dim1_size=orig.dim1_size;
			dim2_size=orig.dim2_size;
			dim3_size=orig.dim3_size;
			
			return *this;
		}

		/** shuffles the array */
		inline void shuffle() { m_array.shuffle(); }

		inline void set_array_name(const char* p_name)
		{
			name=p_name;
		}

		inline void display_array() {}
		inline void display_size() {}

		inline const char* get_array_name() const { return name; }

		/** @return object name */
		inline virtual const char* get_name() const
		{
			return "DynamicArray";
		}

	protected:

		/** underlying array */
		DynArray<T> m_array;

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
#endif /* _DYNAMIC_ARRAY_H_  */
