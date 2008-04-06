/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DYNARRAY_H_
#define _DYNARRAY_H_

/* workaround compile bug in R-modular interface */
#if defined(HAVE_R) && !defined(ScalarReal)
#define ScalarReal      Rf_ScalarReal
#endif

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "base/SGObject.h"

template <class T> class CDynamicArray;

/** dynamic array, i.e. array that can be used like a list or an array.
 * it grows and shrinks dynamically, while elements can be accessed via index
 * performance tuned for simple types as float etc.
 * for hi-level objects only store pointers
 */
template <class T> class CDynamicArray : public CSGObject
{
	public:
		/** constructor
		 *
		 * @param p_resize_granularity resize granularity
		 */
		CDynamicArray(INT p_resize_granularity = 128) : CSGObject()
		{
			this->resize_granularity = p_resize_granularity;

			array = (T*) calloc(p_resize_granularity, sizeof(T));
			ASSERT(array);

			num_elements = p_resize_granularity;
			last_element_idx = -1;
		}

		~CDynamicArray() { free(array); }

		/** set the resize granularity
		 *
		 * @param g new granularity
		 * @return what has been set (minimum is 128)
		 */
		inline INT set_granularity(INT g)
		{
			g=CMath::max(g,128);
			this->resize_granularity = g;
			return g;
		}

		/** get array size (including granularity buffer)
		 *
		 * @return total array size (including granularity buffer)
		 */
		inline INT get_array_size()
		{
			return num_elements;
		}

		/** get number of elements
		 *
		 * @return index of last element
		 */
		inline INT get_num_elements() const
		{
			return last_element_idx+1;
		}

		/** get array element at index
		 *
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline T get_element(INT index) const
		{
			return array[index];
		}

		/** set array element at index
		 *
		 * @param element element to set
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool set_element(T element, INT index)
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
		inline bool insert_element(T element, INT index)
		{
			if (append_element(get_element(last_element_idx)))
			{
				for (INT i=last_element_idx-1; i>index; i--)
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

		/** delete array element at idx
		 * (does not call delete[] or the like)
		 *
		 * @param idx index
		 * @return if deleting was successful
		 */
		inline bool delete_element(INT idx)
		{
			if (idx>=0 && idx<=last_element_idx)
			{
				for (INT i=idx; i<last_element_idx; i++)
					array[i]=array[i+1];

				array[last_element_idx]=0;
				last_element_idx--;

				if ( num_elements - last_element_idx >= resize_granularity)
					resize_array(last_element_idx);

				return true;
			}

			return false;
		}

		/** resize the array
		 *
		 * @param n new size
		 * @return if resizing was successful
		 */
		bool resize_array(INT n)
		{
			INT new_num_elements= ((n/resize_granularity)+1)*resize_granularity;

			T* p= (T*) realloc(array, sizeof(T)*new_num_elements);
			if (p)
			{
				array=p;
				if (new_num_elements > num_elements)
					memset(&array[num_elements], 0, (new_num_elements-num_elements)*sizeof(T));
				else if (n+1<new_num_elements)
					memset(&array[n+1], 0, (new_num_elements-n-1)*sizeof(T));

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
		inline T* get_array()
		{
			return array;
		}

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array new array
		 * @param p_num_elements last element index + 1
		 * @param array_size number of elements in array
		 */
		inline void set_array(T* p_array, INT p_num_elements, INT array_size)
		{
			free(this->array);
			this->array=p_array;
			this->num_elements=array_size;
			this->last_element_idx=p_num_elements-1;
		}

		/** clear the array (with zeros) */
		inline void clear_array()
		{
			if (last_element_idx >= 0)
				memset(array, 0, (last_element_idx+1)*sizeof(T));
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
		inline T operator[](INT index) const
		{
			return array[index];
		}

		/** operator overload for array assignment
		 *
		 * @param orig original array
		 * @return new array
		 */
		CDynamicArray<T>& operator=(CDynamicArray<T>& orig)
		{
			resize_granularity=orig.resize_granularity;
			memcpy(array, orig.array, sizeof(T)*orig.num_elements);
			num_elements=orig.num_elements;
			last_element_idx=orig.last_element_idx;

			return *this;
		}

	protected:
		/** shrink/grow step size */
		INT resize_granularity;

		/** memory for dynamic array */
		T* array;

		/** the number of potentially used elements in array */
		INT num_elements;

		/** the element in the array that has largest index */
		INT last_element_idx;
};
#endif
