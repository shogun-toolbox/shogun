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

#ifndef _DYNAMIC_REFOBJECT_ARRAY_H_
#define _DYNAMIC_REFOBJECT_ARRAY_H_

#include <shogun/base/SGRefObject.h>
#include <shogun/base/DynArray.h>
#include <shogun/base/Parameter.h>

namespace shogun
{
/** @brief Dynamic array class for CRefObject pointers that creates an array
 * that can be used like a list or an array.
 *
 * It grows and shrinks dynamically, while elements can be accessed
 * via index. It only stores CRefObject pointers, which ARE automagically
 * SG_REF'd/deleted.
 *
 * This array is optimized to have very little (storage) overhead
 *
 */
class SGDynamicRefObjectArray : public SGRefObject
{
	public:
		/** default constructor */
		SGDynamicRefObjectArray()
		: SGRefObject(), m_array(), name("Array")
		{
			dim1_size=1;
			dim2_size=1;
			dim3_size=1;
		}

		/** constructor
		 *
		 * @param dim1 dimension 1
		 * @param dim2 dimension 2
		 * @param dim3 dimension 3
		 */
		SGDynamicRefObjectArray(int32_t dim1, int32_t dim2=1, int32_t dim3=1)
		: SGRefObject(), m_array(dim1*dim2*dim3), name("Array")
		{
			dim1_size=dim1;
			dim2_size=dim2;
			dim3_size=dim3;
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_dim1_size dimension 1
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		SGDynamicRefObjectArray(SGRefObject** p_array, int32_t p_dim1_size, bool p_free_array=true, bool p_copy_array=false)
		: SGRefObject(), m_array(p_array, p_dim1_size, p_free_array, p_copy_array), name("Array")
		{
			dim1_size=p_dim1_size;
			dim2_size=1;
			dim3_size=1;
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_dim1_size dimension 1
		 * @param p_dim2_size dimension 2
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		SGDynamicRefObjectArray(SGRefObject** p_array, int32_t p_dim1_size, int32_t p_dim2_size,
						bool p_free_array=true, bool p_copy_array=false)
		: SGRefObject(), m_array(p_array, p_dim1_size*p_dim2_size, p_free_array, p_copy_array), name("Array")
		{
			dim1_size=p_dim1_size;
			dim2_size=p_dim2_size;
			dim3_size=1;
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
		SGDynamicRefObjectArray(SGRefObject** p_array, int32_t p_dim1_size, int32_t p_dim2_size,
						int32_t p_dim3_size, bool p_free_array=true, bool p_copy_array=false)
		: SGRefObject(), m_array(p_array, p_dim1_size*p_dim2_size*p_dim3_size, p_free_array, p_copy_array), name("Array")
		{
			dim1_size=p_dim1_size;
			dim2_size=p_dim2_size;
			dim3_size=p_dim3_size;
		}

		virtual ~SGDynamicRefObjectArray() { unref_all(); }

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
			return m_array.get_num_elements();
		}

		/** get array element at index
		 *
		 * (does NOT do bounds checking)
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline SGRefObject* get_element(int32_t index) const
		{
			SGRefObject* elem=m_array.get_element(index);
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
		inline SGRefObject* element(int32_t idx1, int32_t idx2=0, int32_t idx3=0)
		{
			return get_element(idx1+dim1_size*(idx2+dim2_size*idx3));
		}

		/** get last array element
		 *
		 * @return last array element
		 */
		inline SGRefObject* get_last_element() const
		{
			SGRefObject* e=m_array.get_last_element();
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
		inline SGRefObject* get_element_safe(int32_t index) const
		{
			SGRefObject* e=m_array.get_element_safe(index);
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
		inline bool set_element(SGRefObject* e, int32_t idx1, int32_t idx2=0, int32_t idx3=0)
		{
			int32_t idx = idx1+dim1_size*(idx2+dim2_size*idx3);
			SGRefObject* old=NULL;

			if (idx<get_num_elements())
				old = (SGRefObject*) m_array.get_element(idx);

			bool success=m_array.set_element(e, idx);
			if (success)
			{
				SG_REF(e);
				SG_UNREF(old);
			}

			/* ref before unref to prevent deletion if new=old */
			return success;
		}

		/** insert array element at index
		 *
		 * @param e element to insert
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool insert_element(SGRefObject* e, int32_t index)
		{
			bool success=m_array.insert_element(e, index);
			if (success)
				SG_REF(e);

			return success;
		}

		/** append array element to the end of array
		 *
		 * @param e element to append
		 * @return if setting was successful
		 */
		inline bool append_element(SGRefObject* e)
		{
			bool success=m_array.append_element(e);
			if (success)
				SG_REF(e);

			return success;
		}

		/** STD VECTOR compatible. Append array element to the end
		 *  of array.
		 *
		 * @param e element to append
		 */
		inline void push_back(SGRefObject* e)
		{
			SG_REF(e);
			m_array.push_back(e);
		}

		/** STD VECTOR compatible. Delete array element at the end
		 *  of array.
		 */
		inline void pop_back()
		{
			SGRefObject* e=m_array.back();
			SG_UNREF(e);

			m_array.pop_back();
		}

		/** STD VECTOR compatible. Return array element at the end
		 *  of array.
		 *
		 * @return element at the end of array
		 */
		inline SGRefObject* back() const
		{
			SGRefObject* e=m_array.back();
			SG_REF(e);
			return e;
		}

		/** find first occurence of array element and return its index
		 * or -1 if not available
		 *
		 * @param elem element to search for
		 * @return index of element or -1
		 */
		inline int32_t find_element(SGRefObject* elem) const
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
			SGRefObject* e=m_array.get_element(idx);
			SG_UNREF(e);
			m_array.set_element(NULL, idx);

			return m_array.delete_element(idx);
		}

		/** clear the array (with zeros) */
		inline void clear_array()
		{
			unref_all();
			m_array.clear_array(NULL);
		}

		/** resets the array */
		inline void reset_array()
		{
			unref_all();
			m_array.reset(NULL);
		}

		/** operator overload for array assignment
		 *
		 * @param orig original array
		 * @return new array
		 */
		inline SGDynamicRefObjectArray& operator=(SGDynamicRefObjectArray& orig)
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
		inline SGRefObject** get_array() const { return m_array.get_array(); }

		/** shuffles the array (not thread safe!) */
		inline void shuffle() { m_array.shuffle(); }

		/** shuffles the array with external random state */
		inline void shuffle(CRandom * rand) { m_array.shuffle(rand); }

		/** set array's name
		 *
		 * @param p_name new name
		 */
		inline void set_array_name(const char* p_name)
		{
			name=p_name;
		}

		/** get array's name
		 *
		 * @return array's name
		 */
		inline const char* get_array_name() const { return name; }

		/** @return object name */
		virtual const char* get_name() const
		{ return "DynamicRefObjectArray"; }

	private:
		/** de-reference all elements of this array once */
		inline void unref_all()
		{
			/* SG_UNREF all my elements */
			for (index_t i=0; i<m_array.get_num_elements(); ++i)
			{
				SG_UNREF(*m_array.get_element_ptr(i));
			}
		}

	private:
		/** underlying array */
		DynArray<SGRefObject*> m_array;

		/** dimension 1 */
		int32_t dim1_size;

		/** dimension 2 */
		int32_t dim2_size;

		/** dimension 3 */
		int32_t dim3_size;

		/** array's name */
		const char* name;

};
}
#endif /* _DYNAMIC_REFOBJECT_ARRAY_H_  */
