/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2012 Engeniy Andreev (gsomix)
 */

#ifndef _DYNARRAY_H_
#define _DYNARRAY_H_

#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{
template <class T> class CDynamicArray;

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
	friend class CDynamicObjectArray;
	friend class CCommUlongStringKernel;

	public:
		/** constructor
		 *
		 * @param p_resize_granularity resize granularity
		 * @param tracable
		 */
		DynArray(int32_t p_resize_granularity=128, bool tracable=true)
		{
			resize_granularity=p_resize_granularity;
			free_array=true;
			use_sg_mallocs=tracable;

			if (use_sg_mallocs)
				array=SG_MALLOC(T, p_resize_granularity);
			else
				array=(T*) malloc(size_t(p_resize_granularity)*sizeof(T));

			num_elements=p_resize_granularity;
			current_num_elements=0;
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_array_size array's size
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 * @param tracable
		 */
		DynArray(T* p_array, int32_t p_array_size, bool p_free_array, bool p_copy_array, bool tracable=true)
		{
			resize_granularity=p_array_size;
			free_array=false;
			use_sg_mallocs=tracable;

			array=NULL;
			set_array(p_array, p_array_size, p_array_size, p_free_array, p_copy_array);
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_array_size array's size
		 * @param tracable
		 */
		DynArray(const T* p_array, int32_t p_array_size, bool tracable=true)
		{
			resize_granularity=p_array_size;
			free_array=false;
			use_sg_mallocs=tracable;

			array=NULL;
			set_array(p_array, p_array_size, p_array_size);
		}

		/** destructor */
		virtual ~DynArray()
		{
			if (array!=NULL && free_array)
			{
				if (use_sg_mallocs)
					SG_FREE(array);
				else
					free(array);
			}
		}

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
		inline int32_t get_array_size() const
		{
			return num_elements;
		}

		/** get number of elements
		 *
		 * @return number of elements
		 */
		inline int32_t get_num_elements() const
		{
			return current_num_elements;
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

		/** gets last array element
		 *
		 * @return array element at last index
		 */
		inline T get_last_element() const
		{
			return array[current_num_elements-1];
		}

		/** get array element at index as pointer
		 *
		 * (does NOT do bounds checking)
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline T* get_element_ptr(int32_t index)
		{
			return &array[index];
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
			if (index>=get_num_elements())
			{
				SG_SERROR("array index out of bounds (%d >= %d)\n",
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
			else if (index <= current_num_elements-1)
			{
				array[index]=element;
				return true;
			}
			else if (index < num_elements)
			{
				array[index]=element;
				current_num_elements=index+1;
				return true;
			}
			else
			{
				if (free_array && resize_array(index))
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
			if (append_element(get_element(current_num_elements-1)))
			{
				for (int32_t i=current_num_elements-2; i>index; i--)
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
			return set_element(element, current_num_elements);
		}

		/** STD VECTOR compatible. Append array element to the end
		 *  of array.
		 *
		 * @param element element to append
		 */
		inline void push_back(T element)
		{
			if (get_num_elements() < 0)
				set_element(element, 0);
			else
				set_element(element, get_num_elements());
		}

		/** STD VECTOR compatible. Delete array element at the end
		 *  of array.
		 */
		inline void pop_back()
		{
			if (get_num_elements() <= 0)
				return;

			delete_element(get_num_elements()-1);
		}

		/** STD VECTOR compatible. Return array element at the end
		 *  of array.
		 *
		 * @return element at the end of array
		 */
		inline T back() const
		{
			if (get_num_elements() <= 0)
				return get_element(0);

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
		 * (does not call SG_FREE() or the like)
		 *
		 * @param idx index
		 * @return if deleting was successful
		 */
		inline bool delete_element(int32_t idx)
		{
			if (idx>=0 && idx<=current_num_elements-1)
			{
				for (int32_t i=idx; i<current_num_elements-1; i++)
					array[i]=array[i+1];

				current_num_elements--;

				if (num_elements - current_num_elements - 1
					> resize_granularity)
					resize_array(current_num_elements);

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
		bool resize_array(int32_t n, bool exact_resize=false)
		{
			int32_t new_num_elements=n;

			if (!exact_resize)
			{
				new_num_elements=((n/resize_granularity)+1)*resize_granularity;
			}


			if (use_sg_mallocs)
				array = SG_REALLOC(T, array, num_elements, new_num_elements);
			else
				array = (T*) realloc(array, new_num_elements*sizeof(T));

			//in case of shrinking we must adjust last element idx
			if (n-1<current_num_elements-1)
				current_num_elements=n;

			num_elements=new_num_elements;
			return true;

			return array || new_num_elements==0;
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
			return array;
		}

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array new array
		 * @param p_num_elements last element index + 1
		 * @param p_array_size number of elements in array
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		inline void set_array(T* p_array, int32_t p_num_elements,
				int32_t p_array_size, bool p_free_array, bool p_copy_array)
		{
			if (array!=NULL && free_array)
				SG_FREE(array);

			if (p_copy_array)
			{
				if (use_sg_mallocs)
					array=SG_MALLOC(T, p_array_size);
				else
					array=(T*) malloc(p_array_size*sizeof(T));
				memcpy(array, p_array, p_array_size*sizeof(T));
			}
			else
				array=p_array;

			num_elements=p_array_size;
			current_num_elements=p_num_elements;
			free_array=p_free_array;
		}

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array new array
		 * @param p_num_elements last element index + 1
		 * @param p_array_size number of elements in array
		 */
		inline void set_array(const T* p_array, int32_t p_num_elements,
							  int32_t p_array_size)
		{
			if (array!=NULL && free_array)
				SG_FREE(array);

			if (use_sg_mallocs)
				array=SG_MALLOC(T, p_array_size);
			else
				array=(T*) malloc(p_array_size*sizeof(T));
			memcpy(array, p_array, p_array_size*sizeof(T));

			num_elements=p_array_size;
			current_num_elements=p_num_elements;
			free_array=true;
		}

		/** clear the array (with e.g. zeros) */
		inline void clear_array(T value)
		{
			if (current_num_elements-1 >= 0)
			{
				for (int32_t i=0; i<current_num_elements; i++)
					array[i]=value;
			}
		}

		/** resets the array (as if it was just created), keeps granularity */
		void reset(T value)
		{
			clear_array(value);
			current_num_elements=0;
		}

		/** randomizes the array (not thread safe!) */
		void shuffle()
		{
			for (index_t i=0; i<=current_num_elements-1; ++i)
				CMath::swap(array[i], array[CMath::random(i, current_num_elements-1)]);
		}

		/** randomizes the array with external random state */
		void shuffle(CRandom * rand)
		{
			for (index_t i=0; i<=current_num_elements-1; ++i)
				CMath::swap(array[i], array[rand->random(i, current_num_elements-1)]);
		}

		/** set array with a constant */
		void set_const(const T& const_element)
		{
			for (int32_t i=0; i<num_elements; i++)
				array[i]=const_element;
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

				if (use_sg_mallocs)
					array=SG_MALLOC(T, orig.num_elements);
				else
					array=(T*) malloc(sizeof(T)*orig.num_elements);
			}

			memcpy(array, orig.array, sizeof(T)*orig.num_elements);
			num_elements=orig.num_elements;
			current_num_elements=orig.current_num_elements;

			return *this;
		}

		/** @return object name */
		virtual const char* get_name() const { return "DynArray"; }

	protected:
		/** shrink/grow step size */
		int32_t resize_granularity;

		/** memory for dynamic array */
		T* array;

		/** the number of potentially used elements in array */
		int32_t num_elements;

		/** the number of currently used elements */
		int32_t current_num_elements;

		/** whether SG_MALLOC or just malloc etc shall be used */
		bool use_sg_mallocs;

		/** if array must be freed */
		bool free_array;
};
}
#endif /* _DYNARRAY_H_  */
