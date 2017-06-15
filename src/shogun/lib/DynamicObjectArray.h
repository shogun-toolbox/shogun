/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2011-2016 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DYNAMIC_OBJECT_ARRAY_H_
#define _DYNAMIC_OBJECT_ARRAY_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <vector>
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
		/** default constructor */
		CDynamicObjectArray()
		: CSGObject()
		{
			dim1_size=1;
			dim2_size=1;
			dim3_size=1;

			num_elements = 0;
			free_array = true;
			resize_granularity = 128;
			m_array = std::vector<CSGObject*>(128);
			init();
		}

		/** constructor
		 *
		 * @param dim1 dimension 1
		 * @param dim2 dimension 2
		 * @param dim3 dimension 3
		 */
		CDynamicObjectArray(int32_t dim1, int32_t dim2=1, int32_t dim3=1)
		: CSGObject()
		{
			dim1_size=dim1;
			dim2_size=dim2;
			dim3_size=dim3;

			num_elements = 0;
			free_array = true;
			resize_granularity = dim1 * dim2 * dim3;
			m_array = std::vector<CSGObject*>(dim1 * dim2 * dim3);
			init();
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_dim1_size dimension 1
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		CDynamicObjectArray(CSGObject** p_array, int32_t p_dim1_size, bool p_free_array=true, bool p_copy_array=false)
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
		CDynamicObjectArray(CSGObject** p_array, int32_t p_dim1_size, int32_t p_dim2_size,
						bool p_free_array=true, bool p_copy_array=false)
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
		CDynamicObjectArray(CSGObject** p_array, int32_t p_dim1_size, int32_t p_dim2_size,
						int32_t p_dim3_size, bool p_free_array=true, bool p_copy_array=false)
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

		virtual ~CDynamicObjectArray() { unref_all(); }

		/** set the resize granularity
		 *
		 * @param g new granularity
		 * @return what has been set (minimum is 128)
		 */
		inline int32_t set_granularity(int32_t g)
		{ 
			resize_granularity = g;
			return true;
		}

		/** get array size (including granularity buffer)
		 *
		 * @return total array size (including granularity buffer)
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
		 * (does NOT do bounds checking)
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline CSGObject* get_element(int32_t index) const
		{
			CSGObject* elem=m_array.at(index);
			SG_REF(elem);
			return elem;
		}

		/** get array element at index
		 *
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 3
		 * @return array element at index
		 */
		inline CSGObject* element(int32_t idx1, int32_t idx2=0, int32_t idx3=0)
		{
			return get_element(idx1+dim1_size*(idx2+dim2_size*idx3));
		}

		/** get last array element
		 *
		 * @return last array element
		 */
		inline CSGObject* get_last_element() const
		{
			CSGObject* e=m_array[num_elements - 1];
			SG_REF(e);
			return e;
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
			CSGObject* e=m_array.at(index);
			SG_REF(e);
			return e;
		}

		/** set array element at index
		 *
		 * @param e element to set
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param idx3 index 2
		 * @return if setting was successful
		 */
		inline bool set_element(CSGObject* e, int32_t idx1, int32_t idx2=0, int32_t idx3=0)
		{
			int32_t idx = idx1+dim1_size*(idx2+dim2_size*idx3);
			CSGObject* old=NULL;

			if (idx<get_num_elements())
				old = (CSGObject*) m_array[idx];

			/* ref before unref to prevent deletion if new=old */
			try
 			{
				m_array.insert(m_array.begin() + idx, e);
				SG_REF(e);
				SG_UNREF(old);
				return true;
			}
			catch (const std::bad_alloc&)
			{
				return false;
			}
		}

		/** insert array element at index
		 *
		 * @param e element to insert
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool insert_element(CSGObject* e, int32_t index)
		{
			try
			{
				m_array.insert(m_array.begin() + index, e);
				SG_REF(e);
  
				num_elements++;
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
		inline bool append_element(CSGObject* e)
		{
			if (num_elements < int32_t(m_array.size()))
			{
				m_array[num_elements] = e;
				SG_REF(e);
			}
			else
			{
				m_array.push_back(e);
				SG_REF(e);
			}
			++num_elements;
			return true;
		}

		/** STD VECTOR compatible. Append array element to the end
		 *  of array.
		 *
		 * @param e element to append
		 */
		inline void push_back(CSGObject* e)
		{
			if (num_elements < int32_t(m_array.size()))
			{
				m_array[num_elements] = e;
				SG_REF(e);
			}
			else
			{
				m_array.push_back(e);
				SG_REF(e);
			}
			++num_elements;
		}

		/** STD VECTOR compatible. Delete array element at the end
		 *  of array.
		 */
		inline void pop_back()
		{
			delete_element(num_elements-1);
		}

		/** STD VECTOR compatible. Return array element at the end
		 *  of array.
		 *
		 * @return element at the end of array
		 */
		inline CSGObject* back() const
		{
			CSGObject* e=m_array[num_elements - 1];
			SG_REF(e);
			return e;
		}

		/** find first occurence of array element and return its index
		 * or -1 if not available
		 *
		 * @param elem element to search for
		 * @return index of element or -1
		 */
		inline int32_t find_element(CSGObject* elem) const
		{
			int32_t index = -1;
			for (index_t i = 0; i < num_elements; i++)
			{
				if (m_array[i] == elem)
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
			CSGObject* e = m_array[idx];
			SG_UNREF(e);
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

		/** clear the array (with zeros) */
		inline void clear_array()
		{
			unref_all();
			std::fill(m_array.begin(), m_array.end(), nullptr);
		}

		/** resets the array */
		inline void reset_array()
		{
			unref_all();
			num_elements = 0;
			std::fill(m_array.begin(), m_array.end(), nullptr);
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

			/* copy the vector and number of elements */
			m_array=orig.m_array;
			num_elements=orig.num_elements;
			return *this;
		}

		/** @return underlying array of pointers */
		inline CSGObject** get_array() const 
		{ 
			return (typename std::vector<CSGObject*>::pointer) m_array.data(); 
		}

		/** get array element at index as pointer
		 *
		 * (does NOT do bounds checking)
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline typename std::vector<CSGObject*>::pointer get_element_ptr(int32_t index)
		{
			return &m_array[index];
		}

		/** shuffles the array (not thread safe!) */
		inline void shuffle() 
		{ 
			for (index_t i = 0; i <= num_elements - 1; ++i)
				CMath::swap(
				    m_array[i], m_array[CMath::random(i, num_elements - 1)]); 
		}

		/** shuffles the array with external random state */
		inline void shuffle(CRandom * rand) 
		{ 
			for (index_t i = 0; i <= num_elements - 1; ++i)
				CMath::swap(
				    m_array[i], m_array[rand->random(i, num_elements - 1)]);
		}

 		/** resize array
  		 *
  		 * @param p_array_size the array size we want to have
		 * @param exact_resize resize exactly to the ndim1 * ndim2 * ndim3
		 * @return if resizing was successful
		 */
		inline void
				resize_array(int32_t p_array_size, bool exact_resize = false)
		{
			int32_t new_num_elements = p_array_size;
 			if (!exact_resize)
			{
				new_num_elements =
				    ((new_num_elements / resize_granularity) + 1) *
				    resize_granularity;
			}
			m_array.resize(new_num_elements);
		}

		/** @return object name */
		virtual const char* get_name() const
		{ return "DynamicObjectArray"; }

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
		virtual void save_serializable_pre() throw (ShogunException)
		{
			CSGObject::save_serializable_pre();

			resize_array(get_num_elements(), true);
		}

		virtual CSGObject* clone()
		{
			CDynamicObjectArray* cloned = (CDynamicObjectArray*) CSGObject::clone();
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
			CSGObject** head = m_array.data();
			m_parameters->add_vector(&head, &num_elements, "array", "Memory for dynamic array.");
			SG_ADD(&resize_granularity,
							  "resize_granularity",
							  "shrink/grow step size.", MS_NOT_AVAILABLE);
			SG_ADD(&free_array,
							  "free_array",
							  "whether array must be freed",
							  MS_NOT_AVAILABLE);
			SG_ADD(&dim1_size, "dim1_size", "Dimension 1", MS_NOT_AVAILABLE);
			SG_ADD(&dim2_size, "dim2_size", "Dimension 2", MS_NOT_AVAILABLE);
			SG_ADD(&dim3_size, "dim3_size", "Dimension 3", MS_NOT_AVAILABLE);
		}

		/** de-reference all elements of this array once */
		inline void unref_all()
		{
			/* SG_UNREF all my elements */
			for (index_t i=0; i<get_num_elements(); ++i)
			{
				SG_UNREF(*get_element_ptr(i));
			}
		}

	private:
		/** underlying array */
		std::vector<CSGObject*> m_array;

		/** dimension 1 */
		int32_t dim1_size;

		/** dimension 2 */
		int32_t dim2_size;

		/** dimension 3 */
		int32_t dim3_size;
		
		/** number of elements */
		int32_t num_elements;

		/** if array must be freed */
		bool free_array;

		int32_t resize_granularity;
};
}
#endif /* _DYNAMIC_OBJECT_ARRAY_H_  */
