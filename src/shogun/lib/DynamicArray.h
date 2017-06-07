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

#include <shogun/lib/config.h>

#include <shogun/base/DynArray.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/SGObject.h>
#include <vector>

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

	template <class T> class CDynamicArray : public CSGObject
	{
	public:
		/** default constructor */
		CDynamicArray() : CSGObject()
		{
			dim1_size=1;
			dim2_size=1;
			dim3_size=1;
			num_elements = 0;
			free_array = true;
			resize_granularity = 128;
			m_array = std::vector<T>(resize_granularity);

			init();
		}

		/** constructor
		 *
		 * @param p_dim1_size dimension 1
		 * @param p_dim2_size dimension 2
		 * @param p_dim3_size dimension 3
		 */
		CDynamicArray(
		    int32_t p_dim1_size, int32_t p_dim2_size = 1,
		    int32_t p_dim3_size = 1)
		    : CSGObject()
		{
			dim1_size=p_dim1_size;
			dim2_size=p_dim2_size;
			dim3_size=p_dim3_size;
			num_elements = 0;
			free_array = true;
			resize_granularity = p_dim1_size * p_dim2_size * p_dim3_size;
			m_array = std::vector<T>(resize_granularity);
			init();
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_dim1_size dimension 1
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		CDynamicArray(
		    T* p_array, int32_t p_dim1_size, bool p_free_array,
		    bool p_copy_array)
		    : CSGObject()
		{
			dim1_size=p_dim1_size;
			dim2_size=1;
			dim3_size=1;

			free_array = p_free_array;
			num_elements = p_dim1_size;
			resize_granularity = 128;
			m_array.assign(p_array, p_array + (p_dim1_size));
			init();
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_dim1_size dimension 1
		 * @param p_dim2_size dimension 2
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		CDynamicArray(
		    T* p_array, int32_t p_dim1_size, int32_t p_dim2_size,
		    bool p_free_array, bool p_copy_array)
		    : CSGObject()
		{
			dim1_size=p_dim1_size;
			dim2_size=p_dim2_size;
			dim3_size=1;
			free_array = p_free_array;
			num_elements = p_dim1_size * p_dim2_size;
			resize_granularity = 128;
			m_array.assign(p_array, p_array + (p_dim1_size * p_dim2_size));
			init();
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_dim1_size dimension 1
		 * @param p_dim2_size dimension 2
		 * @param p_dim3_size dimension 3
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		CDynamicArray(
		    T* p_array, int32_t p_dim1_size, int32_t p_dim2_size,
		    int32_t p_dim3_size, bool p_free_array, bool p_copy_array)
		    : CSGObject()
		{
			dim1_size=p_dim1_size;
			dim2_size=p_dim2_size;
			dim3_size=p_dim3_size;
			free_array = p_free_array;
			num_elements = p_dim1_size * p_dim2_size * p_dim3_size;
			resize_granularity = 128;
			m_array.assign(
			    p_array, p_array + (p_dim1_size * p_dim2_size * p_dim3_size));
			init();
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_dim1_size dimension 1
		 * @param p_dim2_size dimension 2
		 * @param p_dim3_size dimension 3
		 */
		CDynamicArray(
		    const T* p_array, int32_t p_dim1_size = 1, int32_t p_dim2_size = 1,
		    int32_t p_dim3_size = 1)
		    : CSGObject()
		{
			dim1_size=p_dim1_size;
			dim2_size=p_dim2_size;
			dim3_size=p_dim3_size;
			num_elements = p_dim1_size * p_dim2_size * p_dim3_size;
			resize_granularity = 128;
			m_array.assign(
			    p_array, p_array + (p_dim1_size * p_dim2_size * p_dim3_size));
			init();
		}

		virtual ~CDynamicArray() {}

		/** set the resize granularity
		 *
		 * @param g new granularity
		 * @return what has been set (minimum is 128)
		 */
		inline int32_t set_granularity(int32_t g)
		{
			resize_granularity = g;
			return resize_granularity;
		}

		/** get array size (including granularity buffer)
		 *
		 * @return total array size
		 */
		inline int32_t get_array_size()
		{
			return m_array.size();
		}

		/** return 2d array size
		 *
		 * @param dim1 dimension 1 will be stored here
		 * @param dim2 dimension 2 will be stored here
		 */
		inline void get_array_size(int32_t& dim1, int32_t& dim2)
		{
			dim1=dim1_size;
			dim2=dim2_size;
		}

		/** return 3d array size
		 *
		 * @param dim1 dimension 1 will be stored here
		 * @param dim2 dimension 2 will be stored here
		 * @param dim3 dimension 3 will be stored here
		 */
		inline void get_array_size(int32_t& dim1, int32_t& dim2, int32_t& dim3)
		{
			dim1=dim1_size;
			dim2=dim2_size;
			dim3=dim3_size;
		}

		/** get dimension 1
		 *
		 * @return dimension 1
		 */
		inline int32_t get_dim1() { return dim1_size; }

		/** get dimension 2
		 *
		 * @return dimension 2
		 */
		inline int32_t get_dim2() { return dim2_size; }

		/** get dimension 3
		 *
		 * @return dimension 3
		 */
		inline int32_t get_dim3() { return dim3_size; }

		/** get number of elements
		 *
		 * @return number of elements
		 */
		inline int32_t get_num_elements() const
		{
			return num_elements;
		}

		/** get array element at index
		 *
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 3
		 * @return array element at index
		 */
		inline typename std::vector<T>::const_reference
		get_element(int32_t idx1, int32_t idx2 = 0, int32_t idx3 = 0) const
		{
			return m_array[idx1 + dim1_size * (idx2 + dim2_size * idx3)];
		}

		/** get array element at index
		 *
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 3
		 * @return array element at index
		 */
		inline typename std::vector<T>::const_reference
		element(int32_t idx1, int32_t idx2 = 0, int32_t idx3 = 0) const
		{
			return get_element(idx1, idx2, idx3);
		}

		/** get array element at index
		 *
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 3
		 * @return array element at index
		 */
		inline typename std::vector<T>::reference
		element(int32_t idx1, int32_t idx2 = 0, int32_t idx3 = 0)
		{
			return m_array[idx1 + dim1_size * (idx2 + dim2_size * idx3)];
		}

		/** get element of given array at given index
		 *
		 * @param p_array another array
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 3
		 * @return array element at index
		 */
		inline typename std::vector<T>::reference
		element(T* p_array, int32_t idx1, int32_t idx2 = 0, int32_t idx3 = 0)
		{
			ASSERT(idx1>=0 && idx1<dim1_size)
			ASSERT(idx2>=0 && idx2<dim2_size)
			ASSERT(idx3>=0 && idx3<dim3_size)
			return p_array[idx1+dim1_size*(idx2+dim2_size*idx3)];
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
		inline typename std::vector<T>::reference element(
		    T* p_array, int32_t idx1, int32_t idx2, int32_t idx3,
		    int32_t p_dim1_size, int32_t p_dim2_size)
		{
			ASSERT(p_dim1_size==dim1_size)
			ASSERT(p_dim2_size==dim2_size)
			ASSERT(idx1>=0 && idx1<p_dim1_size)
			ASSERT(idx2>=0 && idx2<p_dim2_size)
			ASSERT(idx3>=0 && idx3<dim3_size)
			return p_array[idx1+p_dim1_size*(idx2+p_dim2_size*idx3)];
		}

		/** gets last array element
		 *
		 * @return array element at last index
		 */
		inline T get_last_element() const
		{
			return m_array[num_elements - 1];
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
			return m_array.at(index);
		}

		/** set array element at index
		 *
		 * @param e element to set
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 2
		 */
		inline void
		set_element(T e, int32_t idx1, int32_t idx2 = 0, int32_t idx3 = 0)
		{
			m_array[idx1 + dim1_size * (idx2 + dim2_size * idx3)] = e;
		}

		/** insert array element at index
		 *
		 * @param e element to insert
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool insert_element(T e, int32_t index)
		{
			try
			{
				m_array.insert(m_array.begin() + index, e);

				++num_elements;
				return true;
			}
			catch (const std::bad_alloc&)
			{
				return false;
			}
		}

		/** append array element to the end of array
		 *
		 * @param e element to append
		 * @return if setting was successful
		 */
		inline bool append_element(T e)
		{
			if (num_elements < int32_t(m_array.size()))
			{
				m_array[num_elements] = e;
			}
			else
			{
				m_array.push_back(e);
			}
			++num_elements;
			return true;
		}

		/** STD VECTOR compatible. Append array element to the end
		 *  of array.
		 *
		 * @param e element to append
		 */
		inline void push_back(T e)
		{
			if (num_elements < int32_t(m_array.size()))
			{
				m_array[num_elements] = e;
			}
			else
			{
				m_array.push_back(e);
			}
			++num_elements;
		}

		/** STD VECTOR compatible. Delete array element at the end
		 *  of array.
		 */
		inline void pop_back()
		{
			if (get_num_elements() <= 0)
				return;
			delete_element(num_elements - 1);
		}

		/** STD  VECTOR compatible. Return array element at the end
		 *  of array.
		 *
		 * @return element at the end of array
		 */
		inline T back()
		{
			return m_array[num_elements - 1];
		}

		/** find first occurence of array element and return its index
		 * or -1 if not available
		 *
		 * @param e element to search for
		 * @return index of element or -1
		 */
		inline int32_t find_element(T e)
		{
			int32_t index = -1;
			for (index_t i = 0; i < num_elements; i++)
			{
				if (m_array[i] == e)
				{
					index = i;
					break;
				}
			}
			return index;
		}

		/** delete array element at idx
		 * (does not call SG_FREE() or the like)
		 *
		 * @param idx index
		 * @return if deleting was successful
		 */
		inline bool delete_element(int32_t idx)
		{
			int32_t old_length = m_array.size();
			m_array.erase(m_array.begin() + idx);
			if (old_length > int32_t(m_array.size()))
			{
				--num_elements;
				return true;
			}
			else
			{
				return false;
			}
		}

		/** resize array
		 *
		 * @param n new dimension 1
		 * @param exact_resize resize exactly to the ndim1 * ndim2 * ndim3
		 * @return if resizing was successful
		 */
		inline void resize_array(int32_t n, bool exact_resize = false)
		{
			int32_t new_num_elements = n;
			if (!exact_resize)
			{
				new_num_elements =
				    ((new_num_elements / resize_granularity) + 1) *
				    resize_granularity;
			}
			m_array.resize(new_num_elements);
		}

		/** resize array
		 *
		 * @param ndim1 new dimension 1
		 * @param ndim2 new dimension 2
		 * @param exact_resize resize exactly to the ndim1 * ndim2 * ndim3
		 * @return if resizing was successful
		 */
		inline void
		resize_array(int32_t ndim1, int32_t ndim2, bool exact_resize = false)
		{
			dim1_size = ndim1;
			dim2_size = ndim2;
			dim3_size = 1;
			resize_array(ndim1 * ndim2, exact_resize);
		}

		/** resize array
		 *
		 * @param ndim1 new dimension 1
		 * @param ndim2 new dimension 2
		 * @param exact_resize resize exactly to the ndim1 * ndim2 * ndim3
		 * @return if resizing was successful
		 */
		inline void resize_array(
		    int32_t ndim1, int32_t ndim2, int32_t ndim3,
		    bool exact_resize = false)
		{
			dim1_size=ndim1;
			dim2_size=ndim2;
			dim3_size=ndim3;
			resize_array(ndim1 * ndim2 * ndim3, exact_resize);
		}

		/** set array with a constant */
		void set_const(const T& const_element)
		{
			std::fill(m_array.begin(), m_array.end(), const_element);
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
			return m_array.data();
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
			m_array.assign(p_array, p_array + array_size);
			num_elements = p_num_elements;
		}

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array another array
		 * @param dim1 dimension 1
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		inline void set_array(
		    T* p_array, int32_t dim1, bool p_free_array, bool p_copy_array)
		{
			dim1_size=dim1;
			dim2_size=1;
			dim3_size=1;
			if (free_array)
				m_array.clear();

			if (p_copy_array)
			{
				sg_memcpy(m_array.data(), p_array, dim1 * sizeof(T));
			}
			else
				m_array.assign(p_array, p_array + dim1);

			num_elements = dim1;
			free_array = p_free_array;
		}

		/** set the 2d array pointer and free previously allocated memory
		 *
		 * @param p_array another array
		 * @param dim1 dimension 1
		 * @param dim2 dimension 2
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		inline void set_array(
		    T* p_array, int32_t dim1, int32_t dim2, bool p_free_array,
		    bool p_copy_array)
		{
			dim1_size=dim1;
			dim2_size=dim2;
			dim3_size=1;
			int32_t p_array_size = dim1 * dim2;

			set_array(p_array, p_array_size, p_free_array, p_copy_array);
		}

		/** set the 3d array pointer and free previously allocated memory
		 *
		 * @param p_array another array
		 * @param dim1 dimension 1
		 * @param dim2 dimension 2
		 * @param dim3 dimension 3
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		inline void set_array(
		    T* p_array, int32_t dim1, int32_t dim2, int32_t dim3,
		    bool p_free_array, bool p_copy_array)
		{
			dim1_size=dim1;
			dim2_size=dim2;
			dim3_size=dim3;

			int32_t p_array_size = dim1 * dim2 * dim3;

			set_array(p_array, p_array_size, p_free_array, p_copy_array);
		}

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array another array
		 * @param p_size size of another array
		 */
		inline void set_array(const T* p_array, int32_t p_size)
		{
			m_array.assign(p_array, p_array + p_size);
			num_elements = p_size;
		}

		/** clear the array (with e.g. zeros)
		 * @param value value to fill array with
		 */
		inline void clear_array(T value = 0)
		{
			std::fill(m_array.begin(), m_array.end(), value);
		}

		/** resets the array */
		inline void reset_array(T value = 0)
		{
			num_elements = 0;
			std::fill(m_array.begin(), m_array.end(), value);
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
		inline typename std::vector<T>::const_reference
		operator[](int32_t index) const
		{
			return get_element(index);
		}

		/** operator overload for array read-write access
		 *
		 * DOES NOT DO ANY BOUNDS CHECKING
		 *
		 * @param index index
		 * @return element at index
		 */
		inline typename std::vector<T>::reference operator[](int32_t index)
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
			num_elements = orig.num_elements;
			dim1_size=orig.dim1_size;
			dim2_size=orig.dim2_size;
			dim3_size=orig.dim3_size;

			return *this;
		}

		/** shuffles the array (not thread safe!) */
		inline void shuffle()
		{
			for (index_t i = 0; i <= num_elements - 1; ++i)
				CMath::swap(
				    m_array[i], m_array[CMath::random(i, num_elements - 1)]);
		}

		/** shuffles the array with external random state */
		inline void shuffle(CRandom* rand)
		{
			for (index_t i = 0; i <= num_elements - 1; ++i)
				CMath::swap(
				    m_array[i], m_array[rand->random(i, num_elements - 1)]);
		}

		/** display this array */
		inline void display_array()
		{
			if (get_name())
				SG_PRINT("DynamicArray '%s' of size: %dx%dx%d\n", get_name(), dim1_size, dim2_size, dim3_size)
			else
				SG_PRINT("DynamicArray of size: %dx%dx%d\n",dim1_size, dim2_size, dim3_size)

			for (int32_t k=0; k<dim3_size; k++)
				for (int32_t i = 0; i < dim2_size; i++)
				{
					SG_PRINT("element(%d,:,%d) = [ ",i, k)
					for (int32_t j = 0; j < get_num_elements(); j++)
						SG_PRINT("%1.1f,", (float32_t)element(j, i, k))
					SG_PRINT(" ]\n")
				}
		}

		/** display array's size */
		inline void display_size()
		{
			SG_PRINT("DynamicArray of size: %dx%dx%d\n",dim1_size, dim2_size, dim3_size)
		}

		/** @return object name */
		virtual const char* get_name() const
		{
			return "DynamicArray";
		}

		/** Can (optionally) be overridden to pre-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_PRE
		 *  is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		virtual void load_serializable_pre() throw (ShogunException)
		{
			CSGObject::load_serializable_pre();
			m_array.shrink_to_fit();
		}

		/** Can (optionally) be overridden to pre-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::SAVE_SERIALIZABLE_PRE
		 *  is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		virtual void save_serializable_pre() throw (ShogunException)
		{
			CSGObject::save_serializable_pre();
			m_array.shrink_to_fit();
		}

		virtual CSGObject* clone()
		{
			CDynamicArray * cloned = (CDynamicArray*) CSGObject::clone();
			// Since the array vector is registered with
			// current_num_elements as size (see parameter
			// registration) the cloned version has less memory
			// allocated than known to dynarray. We fix this here.
			// cloned->num_elements = cloned->m_array.size();
			return cloned;
		}

	private:

		/** register parameters */
		virtual void init()
		{
			set_generic<T>();
			T* head = m_array.data();
			m_parameters->add_vector(
			    &head, &num_elements, "array", "Memory for dynamic array.");

			SG_ADD(
			    &free_array, "free_array", "whether array must be freed",
			    MS_NOT_AVAILABLE);
			SG_ADD(&dim1_size, "dim1_size", "Dimension 1", MS_NOT_AVAILABLE);
			SG_ADD(&dim2_size, "dim2_size", "Dimension 2", MS_NOT_AVAILABLE);
			SG_ADD(&dim3_size, "dim3_size", "Dimension 3", MS_NOT_AVAILABLE);
		}

	protected:

		/** underlying array */
		std::vector<T> m_array;

		/** number of elements */
		int32_t num_elements;

		/** dimension 1 */
		int32_t dim1_size;

		/** dimension 2 */
		int32_t dim2_size;

		/** dimension 3 */
		int32_t dim3_size;

		/** if array must be freed */
		bool free_array;

		int32_t resize_granularity;
	};

	template <> class CDynamicArray<bool> : public CSGObject
	{
	public:
		/** default constructor */
		CDynamicArray() : CSGObject()
		{
			dim1_size = 1;
			dim2_size = 1;
			dim3_size = 1;
			num_elements = 0;
			array_size = 128;
			free_array = true;
			resize_granularity = 128;
			m_array = SG_MALLOC(bool, resize_granularity);

			init();
		}

		/** constructor
		 *
		 * @param p_dim1_size dimension 1
		 * @param p_dim2_size dimension 2
		 * @param p_dim3_size dimension 3
		 */
		CDynamicArray(
		    int32_t p_dim1_size, int32_t p_dim2_size = 1,
		    int32_t p_dim3_size = 1)
		    : CSGObject()
		{
			dim1_size = p_dim1_size;
			dim2_size = p_dim2_size;
			dim3_size = p_dim3_size;
			num_elements = 0;
			array_size = p_dim1_size * p_dim2_size * p_dim3_size;
			free_array = true;
			resize_granularity = p_dim1_size * p_dim2_size * p_dim3_size;

			m_array = SG_MALLOC(bool, array_size);

			num_elements = 0;

			init();
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_dim1_size dimension 1
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		CDynamicArray(
		    bool* p_array, int32_t p_dim1_size, bool p_free_array,
		    bool p_copy_array)
		    : CSGObject()
		{
			dim1_size = p_dim1_size;
			dim2_size = 1;
			dim3_size = 1;
			free_array = false;
			array_size = p_dim1_size;
			resize_granularity = 128;

			set_array(p_array, p_dim1_size, p_free_array, p_copy_array);
			init();
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_dim1_size dimension 1
		 * @param p_dim2_size dimension 2
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		CDynamicArray(
		    bool* p_array, int32_t p_dim1_size, int32_t p_dim2_size,
		    bool p_free_array, bool p_copy_array)
		    : CSGObject()
		{
			dim1_size = p_dim1_size;
			dim2_size = p_dim2_size;
			dim3_size = 1;
			free_array = false;
			array_size = p_dim1_size * p_dim2_size;
			resize_granularity = 128;

			set_array(p_array, array_size, p_free_array, p_copy_array);
			init();
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_dim1_size dimension 1
		 * @param p_dim2_size dimension 2
		 * @param p_dim3_size dimension 3
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		CDynamicArray(
		    bool* p_array, int32_t p_dim1_size, int32_t p_dim2_size,
		    int32_t p_dim3_size, bool p_free_array, bool p_copy_array)
		    : CSGObject()
		{
			dim1_size = p_dim1_size;
			dim2_size = p_dim2_size;
			dim3_size = p_dim3_size;
			free_array = false;
			array_size = p_dim1_size * p_dim2_size * p_dim3_size;
			resize_granularity = 128;

			set_array(p_array, array_size, p_free_array, p_copy_array);
			init();
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_dim1_size dimension 1
		 * @param p_dim2_size dimension 2
		 * @param p_dim3_size dimension 3
		 */
		CDynamicArray(
		    const bool* p_array, int32_t p_dim1_size = 1,
		    int32_t p_dim2_size = 1, int32_t p_dim3_size = 1)
		    : CSGObject()
		{
			dim1_size = p_dim1_size;
			dim2_size = p_dim2_size;
			dim3_size = p_dim3_size;
			free_array = false;
			array_size = p_dim1_size * p_dim2_size * p_dim3_size;
			resize_granularity = 128;

			set_array(p_array, array_size);
			init();
		}

		virtual ~CDynamicArray()
		{
			if (m_array != NULL && free_array)
			{
				SG_FREE(m_array);
			}
		}

		/** set the resize granularity
		 *
		 * @param g new granularity
		 * @return what has been set (minimum is 128)
		 */
		inline int32_t set_granularity(int32_t g)
		{
			return resize_granularity = g;
		}

		/** get array size (including granularity buffer)
		 *
		 * @return total array size
		 */
		inline int32_t get_array_size()
		{
			return array_size;
		}

		/** return 2d array size
		 *
		 * @param dim1 dimension 1 will be stored here
		 * @param dim2 dimension 2 will be stored here
		 */
		inline void get_array_size(int32_t& dim1, int32_t& dim2)
		{
			dim1 = dim1_size;
			dim2 = dim2_size;
		}

		/** return 3d array size
		 *
		 * @param dim1 dimension 1 will be stored here
		 * @param dim2 dimension 2 will be stored here
		 * @param dim3 dimension 3 will be stored here
		 */
		inline void get_array_size(int32_t& dim1, int32_t& dim2, int32_t& dim3)
		{
			dim1 = dim1_size;
			dim2 = dim2_size;
			dim3 = dim3_size;
		}

		/** get dimension 1
		 *
		 * @return dimension 1
		 */
		inline int32_t get_dim1()
		{
			return dim1_size;
		}

		/** get dimension 2
		 *
		 * @return dimension 2
		 */
		inline int32_t get_dim2()
		{
			return dim2_size;
		}

		/** get dimension 3
		 *
		 * @return dimension 3
		 */
		inline int32_t get_dim3()
		{
			return dim3_size;
		}

		/** get number of elements
		 *
		 * @return number of elements
		 */
		inline int32_t get_num_elements() const
		{
			return num_elements;
		}

		/** get array element at index
		 *
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 3
		 * @return array element at index
		 */
		inline bool&
		get_element(int32_t idx1, int32_t idx2 = 0, int32_t idx3 = 0) const
		{
			return m_array[idx1 + dim1_size * (idx2 + dim2_size * idx3)];
		}

		/** get array element at index
		 *
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 3
		 * @return array element at index
		 */
		inline bool&
		element(int32_t idx1, int32_t idx2 = 0, int32_t idx3 = 0) const
		{
			return get_element(idx1, idx2, idx3);
		}

		/** get array element at index
		 *
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 3
		 * @return array element at index
		 */
		inline bool& element(int32_t idx1, int32_t idx2 = 0, int32_t idx3 = 0)
		{
			return m_array[idx1 + dim1_size * (idx2 + dim2_size * idx3)];
		}

		/** get element of given array at given index
		 *
		 * @param p_array another array
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 3
		 * @return array element at index
		 */
		inline bool&
		element(bool* p_array, int32_t idx1, int32_t idx2 = 0, int32_t idx3 = 0)
		{
			ASSERT(idx1 >= 0 && idx1 < dim1_size)
			ASSERT(idx2 >= 0 && idx2 < dim2_size)
			ASSERT(idx3 >= 0 && idx3 < dim3_size)
			return p_array[idx1 + dim1_size * (idx2 + dim2_size * idx3)];
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
		inline bool& element(
		    bool* p_array, int32_t idx1, int32_t idx2, int32_t idx3,
		    int32_t p_dim1_size, int32_t p_dim2_size)
		{
			ASSERT(p_dim1_size == dim1_size)
			ASSERT(p_dim2_size == dim2_size)
			ASSERT(idx1 >= 0 && idx1 < p_dim1_size)
			ASSERT(idx2 >= 0 && idx2 < p_dim2_size)
			ASSERT(idx3 >= 0 && idx3 < dim3_size)
			return p_array[idx1 + p_dim1_size * (idx2 + p_dim2_size * idx3)];
		}

		/** gets last array element
		 *
		 * @return array element at last index
		 */
		inline bool get_last_element() const
		{
			return m_array[num_elements - 1];
		}

		/** get array element at index
		 *
		 * (does bounds checking)
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline bool get_element_safe(int32_t index) const
		{
			if (index >= get_num_elements())
			{
				SG_SERROR(
				    "array index out of bounds (%d >= %d)\n", index,
				    get_num_elements());
			}
			return m_array[index];
		}

		/** set array element at index
		 *
		 * @param e element to set
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 2
		 * @return if setting was successful
		 */
		inline bool
		set_element(bool e, int32_t idx1, int32_t idx2 = 0, int32_t idx3 = 0)
		{
			int32_t index = idx1 + dim1_size * (idx2 + dim2_size * idx3);
			if (index < 0)
			{
				return false;
			}
			else if (index <= num_elements - 1)
			{
				m_array[index] = e;
				return true;
			}
			else if (index < array_size)
			{
				m_array[index] = e;
				num_elements = index + 1;
				return true;
			}
			else
			{
				if (free_array && resize_array(index + 1))
					return set_element(e, index);
				else
					return false;
			}
		}

		/** insert array element at index
		 *
		 * @param e element to insert
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool insert_element(bool e, int32_t index)
		{
			if (append_element(get_element(num_elements - 1)))
			{
				for (int32_t i = array_size - 2; i > index; i--)
				{
					m_array[i] = m_array[i - 1];
				}
				m_array[index] = e;

				return true;
			}

			return false;
		}

		/** append array element to the end of array
		 *
		 * @param e element to append
		 * @return if setting was successful
		 */
		inline bool append_element(bool e)
		{
			return set_element(e, num_elements);
		}

		/** STD VECTOR compatible. Append array element to the end
		 *  of array.
		 *
		 * @param e element to append
		 */
		inline void push_back(bool e)
		{
			if (get_num_elements() < 0)
				set_element(e, 0);
			else
				set_element(e, get_num_elements());
		}

		/** STD VECTOR compatible. Delete array element at the end
		 *  of array.
		 */
		inline void pop_back()
		{
			if (get_num_elements() <= 0)
				return;

			delete_element(get_num_elements() - 1);
		}

		/** STD  VECTOR compatible. Return array element at the end
		 *  of array.
		 *
		 * @return element at the end of array
		 */
		inline bool back()
		{
			if (get_num_elements() <= 0)
				return get_element(0);

			return get_element(get_num_elements() - 1);
		}

		/** find first occurence of array element and return its index
		 * or -1 if not available
		 *
		 * @param e element to search for
		 * @return index of element or -1
		 */
		inline int32_t find_element(bool e)
		{
			int32_t index = -1;
			int32_t num = get_num_elements();

			for (int32_t i = 0; i < num; i++)
			{
				if (m_array[i] == e)
				{
					index = i;
					break;
				}
			}

			return index;
		}

		/** delete array element at idx
		 * (does not call SG_FREE() or the like)
		 *
		 * @param idx index
		 * @return if deleting was successful
		 */
		inline bool delete_element(int32_t idx)
		{
			if (idx >= 0 && idx <= num_elements - 1)
			{
				for (int32_t i = idx; i < num_elements - 1; i++)
					m_array[i] = m_array[i + 1];

				--num_elements;

				return true;
			}

			return false;
		}

		/** resize the array
		 *
		 * @param n new size
		 * @param exact_resize resize exactly to size n
		 * @return if resizing was successful
		 */
		bool resize_array(int32_t n, bool exact_resize = false)
		{
			int32_t new_num_elements = n;

			if (!exact_resize)
			{
				new_num_elements =
				    ((n / resize_granularity) + 1) * resize_granularity;
			}

			m_array = SG_REALLOC(bool, m_array, array_size, new_num_elements);

			// in case of shrinking we must adjust last element idx
			if (new_num_elements - 1 < num_elements - 1)
				num_elements = new_num_elements;

			array_size = new_num_elements;
			return true;

			return m_array || new_num_elements == 0;
		}

		/** resize array
		 *
		 * @param ndim1 new dimension 1
		 * @param ndim2 new dimension 2
		 * @param exact_resize resize exactly to the ndim1 * ndim2 * ndim3
		 * @return if resizing was successful
		 */
		inline bool
		resize_array(int32_t ndim1, int32_t ndim2, bool exact_resize = false)
		{
			dim1_size = ndim1;
			dim2_size = ndim2;
			int32_t new_num_elements = ndim1 * ndim2;
			return resize_array(new_num_elements, exact_resize);
		}

		/** resize array
		 *
		 * @param ndim1 new dimension 1
		 * @param ndim2 new dimension 2
		 * @param ndim3 new dimension 3
		 * @param exact_resize resize exactly to the ndim1 * ndim2 * ndim3
		 * @return if resizing was successful
		 */
		inline bool resize_array(
		    int32_t ndim1, int32_t ndim2, int32_t ndim3,
		    bool exact_resize = false)
		{
			dim1_size = ndim1;
			dim2_size = ndim2;
			dim3_size = ndim3;
			int32_t new_num_elements = ndim1 * ndim2 * ndim3;
			return resize_array(new_num_elements, exact_resize);
		}

		/** set array with a constant */
		void set_const(const bool& const_element)
		{
			for (int32_t i = 0; i < array_size; i++)
				m_array[i] = const_element;
		}

		/** get the array
		 * call get_array just before messing with it DO NOT call any
		 * [],resize/delete functions after get_array(), the pointer may
		 * become invalid !
		 *
		 * @return the array
		 */
		inline bool* get_array()
		{
			return m_array;
		}

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array new array
		 * @param p_num_elements last element index + 1
		 * @param array_size number of elements in array
		 */
		inline void
		set_array(bool* p_array, int32_t p_num_elements, int32_t p_array_size)
		{
			if (m_array != NULL && free_array)
				SG_FREE(m_array);

			m_array = SG_MALLOC(bool, p_array_size);

			sg_memcpy(m_array, p_array, p_array_size * sizeof(bool));

			array_size = p_array_size;
			num_elements = p_num_elements;
			free_array = true;
		}

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array another array
		 * @param dim1 dimension 1
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		inline void set_array(
		    bool* p_array, int32_t dim1, bool p_free_array, bool p_copy_array)
		{
			dim1_size = dim1;
			dim2_size = 1;
			dim3_size = 1;
			if (m_array != NULL && free_array)
				SG_FREE(m_array);

			if (p_copy_array)
			{
				m_array = SG_MALLOC(bool, dim1);
				sg_memcpy(m_array, p_array, dim1 * sizeof(bool));
			}
			else
				m_array = p_array;

			array_size = dim1;
			num_elements = dim1;
			free_array = p_free_array;
		}

		/** set the 2d array pointer and free previously allocated memory
		 *
		 * @param p_array another array
		 * @param dim1 dimension 1
		 * @param dim2 dimension 2
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		inline void set_array(
		    bool* p_array, int32_t dim1, int32_t dim2, bool p_free_array,
		    bool p_copy_array)
		{
			dim1_size = dim1;
			dim2_size = dim2;
			dim3_size = 1;
			int32_t p_array_size = dim1 * dim2;

			set_array(p_array, p_array_size, p_free_array, p_copy_array);
		}

		/** set the 3d array pointer and free previously allocated memory
		 *
		 * @param p_array another array
		 * @param dim1 dimension 1
		 * @param dim2 dimension 2
		 * @param dim3 dimension 3
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		inline void set_array(
		    bool* p_array, int32_t dim1, int32_t dim2, int32_t dim3,
		    bool p_free_array, bool p_copy_array)
		{
			dim1_size = dim1;
			dim2_size = dim2;
			dim3_size = dim3;
			int32_t p_array_size = dim1 * dim2 * dim3;

			set_array(p_array, p_array_size, p_free_array, p_copy_array);
		}

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array new array
		 * @param p_size size of another array
		 */
		inline void set_array(const bool* p_array, int32_t p_size)
		{
			if (m_array != NULL && free_array)
				SG_FREE(m_array);

			m_array = SG_MALLOC(bool, p_size);

			sg_memcpy(m_array, p_array, p_size * sizeof(bool));

			array_size = p_size;
			num_elements = p_size;
			free_array = true;
		}

		/** clear the array (with e.g. zeros)
		 * @param value value to fill array with
		 */
		inline void clear_array(bool value = false)
		{
			if (num_elements - 1 >= 0)
			{
				for (int32_t i = 0; i < num_elements; i++)
					m_array[i] = value;
			}
		}

		/** resets the array */
		inline void reset_array(bool value = false)
		{
			clear_array(value);
			num_elements = 0;
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
		inline const bool& operator[](int32_t index) const
		{
			return get_element(index);
		}

		/** operator overload for array read-write access
		 *
		 * DOES NOT DO ANY BOUNDS CHECKING
		 *
		 * @param index index
		 * @return element at index
		 */
		inline bool& operator[](int32_t index)
		{
			return element(index);
		}

		/** operator overload for array assignment
		 *
		 * @param orig original array
		 * @return new array
		 */
		inline CDynamicArray<bool>& operator=(CDynamicArray<bool>& orig)
		{
			m_array = orig.m_array;
			num_elements = orig.num_elements;
			dim1_size = orig.dim1_size;
			dim2_size = orig.dim2_size;
			dim3_size = orig.dim3_size;

			return *this;
		}

		/** shuffles the array (not thread safe!) */
		inline void shuffle()
		{
			for (index_t i = 0; i <= num_elements - 1; ++i)
				CMath::swap(
				    m_array[i], m_array[CMath::random(i, num_elements - 1)]);
		}

		/** shuffles the array with external random state */
		inline void shuffle(CRandom* rand)
		{
			for (index_t i = 0; i <= num_elements - 1; ++i)
				CMath::swap(
				    m_array[i], m_array[rand->random(i, num_elements - 1)]);
		}

		/** display this array */
		inline void display_array()
		{
			if (get_name())
				SG_PRINT(
				    "DynamicArray '%s' of size: %dx%dx%d\n", get_name(),
				    dim1_size, dim2_size, dim3_size)
			else
				SG_PRINT(
				    "DynamicArray of size: %dx%dx%d\n", dim1_size, dim2_size,
				    dim3_size)

			for (int32_t k = 0; k < dim3_size; k++)
				for (int32_t j = 0; j < dim2_size; j++)
				{
					SG_PRINT("element(%d,:,%d) = [ ", j, k)
					for (int32_t i = 0; i < get_num_elements(); i++)
						SG_PRINT("%1.1f,", (float32_t)element(i, j, k))
					SG_PRINT(" ]\n")
				}
		}

		/** display array's size */
		inline void display_size()
		{
			SG_PRINT(
			    "DynamicArray of size: %dx%dx%d\n", dim1_size, dim2_size,
			    dim3_size)
		}

		/** @return object name */
		virtual const char* get_name() const
		{
			return "DynamicArray";
		}

		/** Can (optionally) be overridden to pre-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::LOAD_SERIALIZABLE_PRE
		 *  is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		virtual void load_serializable_pre() throw(ShogunException)
		{
			CSGObject::load_serializable_pre();

			resize_array(get_num_elements(), true);
		}

		/** Can (optionally) be overridden to pre-initialize some member
		 *  variables which are not PARAMETER::ADD'ed.  Make sure that at
		 *  first the overridden method BASE_CLASS::SAVE_SERIALIZABLE_PRE
		 *  is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		virtual void save_serializable_pre() throw(ShogunException)
		{
			CSGObject::save_serializable_pre();
			resize_array(get_num_elements(), true);
		}

		virtual CSGObject* clone()
		{
			CDynamicArray* cloned = (CDynamicArray*)CSGObject::clone();
			// Since the array vector is registered with
			// current_num_elements as size (see parameter
			// registration) the cloned version has less memory
			// allocated than known to dynarray. We fix this here.
			// cloned->num_elements = cloned->array_size;
			return cloned;
		}

	private:
		/** register parameters */
		virtual void init()
		{
			set_generic<bool>();
			m_parameters->add_vector(
			    &m_array, &num_elements, "array", "Memory for dynamic array.");

			SG_ADD(
			    &free_array, "free_array", "whether array must be freed",
			    MS_NOT_AVAILABLE);
			SG_ADD(&dim1_size, "dim1_size", "Dimension 1", MS_NOT_AVAILABLE);
			SG_ADD(&dim2_size, "dim2_size", "Dimension 2", MS_NOT_AVAILABLE);
			SG_ADD(&dim3_size, "dim3_size", "Dimension 3", MS_NOT_AVAILABLE);
		}

	protected:
		/** underlying array */
		bool* m_array;

		/** number of currently used elements */
		int32_t num_elements;

		/** dimension 1 */
		int32_t dim1_size;

		/** dimension 2 */
		int32_t dim2_size;

		/** dimension 3 */
		int32_t dim3_size;

		/**size of array */
		int32_t array_size;

		/** if array must be freed */
		bool free_array;

		int32_t resize_granularity;
};
}
#endif /* _DYNAMIC_ARRAY_H_  */
