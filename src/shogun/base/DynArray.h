/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DYNARRAY_H_
#define _DYNARRAY_H_

#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{
extern IO* sg_io;

template <class T> class CDynamicArray;
template <class T> class CDynamicObjectArray;

/** @brief Template Dynamic array class that creates an array that can
 * be used like a list or an array.
 *
 * It grows and shrinks dynamically, while elements can be accessed
 * via index.  It is performance tuned for simple types like float
 * etc. and for hi-level objects only stores pointers, which are not
 * automagically SG_REF'd/deleted.
 */
template <class T> class DynArray
{
	template<class U> friend class CDynamicArray;
	template<class U> friend class CDynamicObjectArray;
	friend class CCommUlongStringKernel;

	public:
		/** constructor
		 *
		 * @param p_resize_granularity resize granularity
		 */
		DynArray(int32_t p_resize_granularity=128)
		{
			this->resize_granularity=p_resize_granularity;

			array=(T*)SG_CALLOC(p_resize_granularity, sizeof(T));

			num_elements=p_resize_granularity;
			last_element_idx=-1;
		}

		virtual ~DynArray(void)
		{ SG_FREE(array); }

		/** set the resize granularity
		 *
		 * @param g new granularity
		 * @return what has been set (minimum is 128)
		 */
		inline int32_t set_granularity(int32_t g)
		{
			g=CMath::max(g,128);
			this->resize_granularity = g;
			return g;
		}

		/** get array size (including granularity buffer)
		 *
		 * @return total array size (including granularity buffer)
		 */
		inline int32_t get_array_size(void) const
		{
			return num_elements;
		}

		/** get number of elements
		 *
		 * @return number of elements
		 */
		inline int32_t get_num_elements(void) const
		{
			return last_element_idx+1;
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
			return array[index];
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
			IO* io = sg_io;

			if (index>=get_num_elements())
			{
				SG_ERROR("array index out of bounds (%d >= %d)\n",
						 index, get_num_elements());
			}
			return array[index];
		}

		/** set array element at index
		 *
		 * @param element element to set
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool set_element(T element, int32_t index)
		{
			if (index < 0)
			{
				return false;
			}
			else if (index <= last_element_idx)
			{
				array[index]=element;
				return true;
			}
			else if (index < num_elements)
			{
				array[index]=element;
				last_element_idx=index;
				return true;
			}
			else
			{
				if (resize_array(index))
					return set_element(element, index);
				else
					return false;
			}
		}

		/** insert array element at index
		 *
		 * @param element element to insert
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool insert_element(T element, int32_t index)
		{
			if (append_element(get_element(last_element_idx)))
			{
				for (int32_t i=last_element_idx-1; i>index; i--)
				{
					array[i]=array[i-1];
				}
				array[index]=element;

				return true;
			}

			return false;
		}

		/** append array element to the end of array
		 *
		 * @param element element to append
		 * @return if setting was successful
		 */
		inline bool append_element(T element)
		{
			return set_element(element, last_element_idx+1);
		}

		/** ::STD::VECTOR compatible. Append array element to the end
		 *  of array.
		 *
		 * @param element element to append
		 */
		inline void push_back(T element)
		{
			if (get_num_elements() < 0) set_element(element, 0);
			else set_element(element, get_num_elements());
		}

		/** ::STD::VECTOR compatible. Delete array element at the end
		 *  of array.
		 */
		inline void pop_back(void)
		{
			if (get_num_elements() <= 0) return;
			delete_element(get_num_elements()-1);
		}

		/** ::STD::VECTOR compatible. Return array element at the end
		 *  of array.
		 *
		 * @return element at the end of array
		 */
		inline T back(void) const
		{
			if (get_num_elements() <= 0) return get_element(0);
			return get_element(get_num_elements()-1);
		}

		/** find first occurence of array element and return its index
		 * or -1 if not available
		 *
		 * @param element element to search for
		 * @return index of element or -1
		 */
		int32_t find_element(T element) const
		{
			int32_t idx=-1;
			int32_t num=get_num_elements();

			for (int32_t i=0; i<num; i++)
			{
				if (array[i] == element)
				{
					idx=i;
					break;
				}
			}

			return idx;
		}

		/** delete array element at idx
		 * (does not call delete[] or the like)
		 *
		 * @param idx index
		 * @return if deleting was successful
		 */
		inline bool delete_element(int32_t idx)
		{
			if (idx>=0 && idx<=last_element_idx)
			{
				for (int32_t i=idx; i<last_element_idx; i++)
					array[i]=array[i+1];

				array[last_element_idx]=0;
				last_element_idx--;

				if (num_elements - last_element_idx
					> resize_granularity)
					resize_array(last_element_idx+1);

				return true;
			}

			return false;
		}

		/** resize the array
		 *
		 * @param n new size
		 * @return if resizing was successful
		 */
		bool resize_array(int32_t n)
		{
			int32_t new_num_elements= ((n/resize_granularity)+1)
				*resize_granularity;

			T* p= (T*) SG_REALLOC(array, sizeof(T)*new_num_elements);
			if (p)
			{
				array=p;
				if (new_num_elements > num_elements)
					memset(&array[num_elements], 0,
						   (new_num_elements-num_elements)*sizeof(T));
				else if (n+1<new_num_elements)
					memset(&array[n+1], 0,
						   (new_num_elements-n-1)*sizeof(T));

				//in case of shrinking we must adjust last element idx
				if (n-1<last_element_idx)
					last_element_idx=n-1;

				num_elements=new_num_elements;
				return true;
			}
			else
				return false;
		}

		/** get the array
		 * call get_array just before messing with it DO NOT call any
		 * [],resize/delete functions after get_array(), the pointer may
		 * become invalid !
		 *
		 * @return the array
		 */
		inline T* get_array(void)
		{
			return array;
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
			SG_FREE(this->array);
			this->array=p_array;
			this->num_elements=array_size;
			this->last_element_idx=p_num_elements-1;
		}

		/** clear the array (with zeros) */
		inline void clear_array(void)
		{
			if (last_element_idx >= 0)
				memset(array, 0, (last_element_idx+1)*sizeof(T));
		}

		/** randomizes the array */
		void shuffle()
		{
			for (index_t i=0; i<=last_element_idx; ++i)
				CMath::swap(array[i], array[CMath::random(i, last_element_idx)]);
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
			return array[index];
		}

		/** operator overload for array assignment.
		 * Left array is resized if needed.
		 *
		 * @param orig original array
		 * @return new array
		 */
		DynArray<T>& operator=(DynArray<T>& orig)
		{
			resize_granularity=orig.resize_granularity;

			/* check if orig array is larger than current, create new array */
			if (orig.num_elements>num_elements)
			{
				SG_FREE(array);
				array=(T*)SG_MALLOC(orig.num_elements*sizeof(T));
			}

			memcpy(array, orig.array, sizeof(T)*orig.num_elements);
			num_elements=orig.num_elements;
			last_element_idx=orig.last_element_idx;

			return *this;
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "DynArray"; }

	protected:
		/** shrink/grow step size */
		int32_t resize_granularity;

		/** memory for dynamic array */
		T* array;

		/** the number of potentially used elements in array */
		int32_t num_elements;

		/** the element in the array that has largest index */
		int32_t last_element_idx;
};
}
#endif /* _DYNARRAY_H_  */
