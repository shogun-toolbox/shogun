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

#include <shogun/base/SGObject.h>
#include <shogun/base/DynArray.h>
#include <shogun/base/Parameter.h>

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
 * T specifies the type of the pointers
 */
template<class T>class CDynamicObjectArray :public CSGObject
{
	DynArray<T*> m_array;

	public:
		/** constructor
		 *
		 * @param p_resize_granularity resize granularity
		 */
		CDynamicObjectArray(int32_t p_resize_granularity=128)
		: CSGObject()
		{
			CSGObject*** casted_array=(CSGObject***)&m_array.array;

			m_parameters->add_vector(casted_array, &m_array.num_elements, "array",
					"Memory for dynamic array.");
			m_parameters->add(&m_array.last_element_idx, "last_element_idx",
					"Element with largest index.");
			m_parameters->add(&m_array.resize_granularity, "resize_granularity",
					"shrink/grow step size.");
		}

		virtual ~CDynamicObjectArray() { unref_all(); }

		/** set the resize granularity
		 *
		 * @param g new granularity
		 * @return what has been set (minimum is 128)
		 */
		inline int32_t set_granularity(int32_t g)
		{ return m_array.set_granularity(g); }

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
		inline T* get_element(int32_t index) const
		{
			T* element=m_array.get_element(index);
			CSGObject* casted=cast_to_sgobject(element);
			SG_REF(casted);
			return element;
		}

		/** get array element at index
		 *
		 * (does bounds checking)
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline T* get_element_safe(int32_t index) const
		{
			T* element=m_array.get_element_safe(index);
			CSGObject* casted=(CSGObject*)element;
			SG_REF(casted);
			return element;
		}

		/** set array element at index
		 *
		 * @param element element to set
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool set_element(T* element, int32_t index)
		{
			CSGObject* casted=cast_to_sgobject(element);
			CSGObject* old=(CSGObject*)m_array.get_element(index);

			bool success=m_array.set_element(element, index);
			if (success)
			{
				SG_REF(casted);
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
		inline bool insert_element(T* element, int32_t index)
		{
			CSGObject* casted=cast_to_sgobject(element);
			bool success=m_array.insert_element(element, index);
			if (success)
				SG_REF(casted);

			return success;
		}

		/** append array element to the end of array
		 *
		 * @param element element to append
		 * @return if setting was successful
		 */
		inline bool append_element(T* element)
		{
			CSGObject* casted=cast_to_sgobject(element);
			bool success=m_array.append_element(element);
			if (success)
				SG_REF(casted);

			return success;
		}

		/** STD VECTOR compatible. Append array element to the end
		 *  of array.
		 *
		 * @param element element to append
		 */
		inline void push_back(T* element)
		{
			CSGObject* casted=cast_to_sgobject(element);
			SG_REF(casted);
			m_array.push_back(element);
		}

		/** STD VECTOR compatible. Delete array element at the end
		 *  of array.
		 */
		inline void pop_back()
		{
			CSGObject* element=(CSGObject*)m_array.back();
			SG_UNREF(element);

			m_array.pop_back();
		}

		/** STD VECTOR compatible. Return array element at the end
		 *  of array.
		 *
		 * @return element at the end of array
		 */
		inline T* back() const
		{
			T* element=m_array.back();
			CSGObject* casted=(CSGObject*)element;
			SG_REF(casted);
			return element;
		}

		/** find first occurence of array element and return its index
		 * or -1 if not available
		 *
		 * @param element element to search for
		 * @return index of element or -1
		 */
		inline int32_t find_element(T* element) const
		{
			return m_array.find_element(element);
		}

		/** delete array element at idx
		 * (does not call SG_FREE() or the like)
		 *
		 * @param idx index
		 * @return if deleting was successful
		 */
		inline bool delete_element(int32_t idx)
		{
			CSGObject* element=(CSGObject*)m_array.get_element(idx);
			SG_UNREF(element);

			return m_array.delete_element(idx);
		}

		/** clear the array (with zeros) */
		inline void clear_array()
		{
			unref_all();
			m_array.clear_array();
		}

		/** operator overload for array assignment
		 *
		 * @param orig original array
		 * @return new array
		 */
		inline CDynamicObjectArray<T>& operator=(CDynamicObjectArray<T>& orig)
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
		inline T** get_array() const { return m_array.get_array(); }

		/** shuffles the array */
		inline void shuffle() { m_array.shuffle(); }

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
				CSGObject* element=(CSGObject*)m_array.get_element(i);
				SG_UNREF(element);
			}
		}

		/** Dynamically casts the given element to a CSGObject, if non-NULL.
		 * Used for all method that have some input elements that are to be put
		 * into the array */
		inline CSGObject* cast_to_sgobject(T* element) const
		{
			if (!element)
				return NULL;

			CSGObject* casted=dynamic_cast<CSGObject*>(element);

			if (!casted)
			{
				SG_ERROR("Generic type of CDynamicObjectArray is not of type "
						"CSGObject!\n");
			}

			return casted;
		}
};
}
#endif /* _DYNAMIC_OBJECT_ARRAY_H_  */
