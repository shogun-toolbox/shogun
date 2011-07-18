/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg, Gunnar Raetsch, Andre Noll
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _ARRAY_H_
#define _ARRAY_H_

//#define ARRAY_STATISTICS

//#define ARRAY_ASSERT(x) {if ((x)==0) {*((int*)0)=0;}}
//#define ARRAY_ASSERT(x) ASSERT(x)
#define ARRAY_ASSERT(x)

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>

namespace shogun
{
#ifdef ARRAY_STATISTICS
struct array_statistics {
	int32_t const_element;
	int32_t element;
	int32_t set_element;
	int32_t get_element;
	int32_t operator_overload;
	int32_t const_operator_overload;
	int32_t set_array;
	int32_t get_array;
	int32_t resize_array;
	int32_t array_element;
};

#define DECLARE_ARRAY_STATISTICS struct array_statistics as
#define INIT_ARRAY_STATISTICS memset(&as, 0, sizeof(as))
#define PRINT_ARRAY_STATISTICS \
	SG_DEBUG("access statistics:\n" \
			"const element    %i\n" \
			"element    %i\n" \
			"set_element    %i\n" \
			"get_element    %i\n" \
			"operator_overload[]    %i\n" \
			"const_operator_overload[]    %i\n" \
			"set_array    %i\n" \
			"get_array    %i\n" \
			"resize_array    %i\n" \
			"array_element    %i\n", \
			as.const_element, \
			as.element, \
			as.set_element, \
			as.get_element, \
			as.operator_overload, \
			as.const_operator_overload, \
			as.set_array, \
			as.get_array, \
			as.resize_array, \
			as.array_element \
);

#define INCREMENT_ARRAY_STATISTICS_VALUE(_val_) ((CArray<T>*)this)->as._val_++

#else /* ARRAY_STATISTICS */
#define DECLARE_ARRAY_STATISTICS
#define INIT_ARRAY_STATISTICS
#define PRINT_ARRAY_STATISTICS
#define INCREMENT_ARRAY_STATISTICS_VALUE(_val_)
#endif /* ARRAY_STATISTICS */

/** @brief Template class Array implements a dense one dimensional array.
 *
 * Note that depending on compile options everything will be inlined, such that
 * this is as high performance array implementation \b without error checking.
 *
 * */
template <class T> class CArray : public CSGObject
{
	public:
		/** constructor
		 *
		 * @param initial_size initial size of array
		 */
		CArray(int32_t initial_size = 1)
		: CSGObject(), free_array(true), name("Array")
		{
			INIT_ARRAY_STATISTICS;
			array_size = initial_size;
			array = (T*) calloc(array_size, sizeof(T));
			ARRAY_ASSERT(array);
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_array_size size of another array
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		CArray(T* p_array, int32_t p_array_size, bool p_free_array=true,
			bool p_copy_array=false)
		: CSGObject(), array(NULL), free_array(false), name("Array")
		{
			INIT_ARRAY_STATISTICS;
			set_array(p_array, p_array_size, p_free_array, p_copy_array);
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param p_array_size size of another array
		 */
		CArray(const T* p_array, int32_t p_array_size)
		: CSGObject(), array(NULL), free_array(false), name("Array")
		{
			INIT_ARRAY_STATISTICS;
			set_array(p_array, p_array_size);
		}

		virtual ~CArray()
		{
			//SG_DEBUG( "destroying CArray array '%s' of size %i\n", name? name : "unnamed", array_size);
			PRINT_ARRAY_STATISTICS;
			SG_FREE(array);
		}

		/** get name
		 *
		 * @return name
		 */
		inline virtual const char* get_name() const { return "Array"; }

		/** get array name
		 *
		 * @return name
		 */
		inline virtual const char* get_array_name() const { return name; }

		/** set name
		 *
		 * @param p_name new name
		 */
		inline void set_array_name(const char* p_name)
		{
			name = p_name;
		}

		/** get array size (including granularity buffer)
		 *
		 * @return total array size
		 */
		inline int32_t get_array_size() const
		{
			return array_size;
		}

		/** get array size (including granularity buffer)
		 *
		 * @return total array size
		 */
		inline int32_t get_dim1()
		{
			return array_size;
		}

		/** zero array */
		inline void zero()
		{
			for (int32_t i=0; i< array_size; i++)
				array[i]=0;
		}

		/** set array with a constant */
		inline void set_const(T const_elem)
		{
			for (int32_t i=0; i< array_size; i++)
				array[i]=const_elem ;
		}

		/** get array element at index
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline const T& get_element(int32_t index) const
		{
			ARRAY_ASSERT(array && (index>=0) && (index<array_size));
			INCREMENT_ARRAY_STATISTICS_VALUE(get_element);
			return array[index];
		}

		/** set array element at index 'index' return false in case of trouble
		 *
		 * @param p_element array element to set
		 * @param index index
		 * @return if setting was successful
		 */
		inline bool set_element(const T& p_element, int32_t index)
		{
			ARRAY_ASSERT(array && (index>=0) && (index<array_size));
			INCREMENT_ARRAY_STATISTICS_VALUE(set_element);
			array[index]=p_element;
			return true;
		}

		/** get element at given index
		 *
		 * @param idx1 index
		 * @return element at given index
		 */
		inline const T& element(int32_t idx1) const
		{
			INCREMENT_ARRAY_STATISTICS_VALUE(const_element);
			return get_element(idx1);
		}

		/** get element at given index
		 *
		 * @param index index
		 * @return element at given index
		 */
		inline T& element(int32_t index)
		{
			ARRAY_ASSERT(array);
			ARRAY_ASSERT(index>=0);
			ARRAY_ASSERT(index<array_size);
			INCREMENT_ARRAY_STATISTICS_VALUE(element);
			return array[index];
		}

		/** get element of given array at given index
		 *
		 * @param p_array another array
		 * @param index index
		 * @return element of given array at given index
		 */
		inline T& element(T* p_array, int32_t index)
		{
			ARRAY_ASSERT(array && (index>=0) && (index<array_size));
			ARRAY_ASSERT(array == p_array);
			INCREMENT_ARRAY_STATISTICS_VALUE(array_element);
			return p_array[index];
		}

		/** resize array
		 *
		 * @param n new size
		 * @return if resizing was successful
		 */
		bool resize_array(int32_t n)
		{
			INCREMENT_ARRAY_STATISTICS_VALUE(resize_array);
			ARRAY_ASSERT(free_array);

			T* p= (T*) SG_REALLOC(array, sizeof(T)*n);
			if (!p)
				return false;
			array=p;
			if (n > array_size)
				memset(&array[array_size], 0, (n-array_size)*sizeof(T));
			array_size=n;
			return true;
		}

		/** call get_array just before messing with it DO NOT call any
		 *  [],resize/delete functions after get_array(), the pointer
		 *  may become invalid!
		 *
		 *  @return the array
		 */
		inline T* get_array()
		{
			INCREMENT_ARRAY_STATISTICS_VALUE(get_array);
			return array;
		}

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array another array
		 * @param p_array_size size of another array
		 * @param p_free_array if array must be freed
		 * @param copy_array if array must be copied
		 */
		inline void set_array(T* p_array, int32_t p_array_size, bool p_free_array=true,
				bool copy_array=false)
		{
			INCREMENT_ARRAY_STATISTICS_VALUE(set_array);
			SG_FREE(this->array);
			if (copy_array)
			{
				this->array=(T*)SG_MALLOC(p_array_size*sizeof(T));
				memcpy(this->array, p_array, p_array_size*sizeof(T));
			}
			else
				this->array=p_array;
			this->array_size=p_array_size;
			this->free_array=p_free_array;
		}

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array another array
		 * @param p_array_size size of another array
		 */
		inline void set_array(const T* p_array, int32_t p_array_size)
		{
			INCREMENT_ARRAY_STATISTICS_VALUE(set_array);
			SG_FREE(this->array);
			this->array=(T*)SG_MALLOC(p_array_size*sizeof(T));
			memcpy(this->array, p_array, p_array_size*sizeof(T));
			this->array_size=p_array_size;
			this->free_array=true;
		}

		/** clear the array (with zeros) */
		inline void clear_array()
		{
			memset(array, 0, array_size*sizeof(T));
		}


		/** operator overload for array read only access
		 * use set_element() for write access (will also make the array dynamically grow)
		 *
		 * DOES NOT DO ANY BOUNDS CHECKING
		 *
		 * @param index
		 * @return element at index
		 */
		inline const T& operator[](int32_t index) const
		{
			INCREMENT_ARRAY_STATISTICS_VALUE(const_operator_overload);
			return array[index];
		}

		/** operator overload for array read only access
		 *
		 * DOES NOT DO ANY BOUNDS CHECKING
		 *
		 * @param index
		 * @return element at index
		 */
		inline T& operator[](int32_t index)
		{
			INCREMENT_ARRAY_STATISTICS_VALUE(operator_overload);
			return element(index);
		}

		/** operator overload for array assignment
		 *
		 * @param orig original array
		 * @return new array
		 */
		CArray<T>& operator=(const CArray<T>& orig)
		{
			memcpy(array, orig.array, sizeof(T)*orig.array_size);
			array_size=orig.array_size;

			return *this;
		}

		/** display array size */
		void display_size() const
		{
			SG_PRINT( "Array '%s' of size: %d\n", name? name : "unnamed",
					array_size);
		}

		/** display array */
		void display_array() const
		{
			display_size();
			for (int32_t i=0; i<array_size; i++)
				SG_PRINT("%1.1f,", (float32_t)array[i]);
			SG_PRINT("\n");
		}

	protected:
		/** memory for dynamic array */
		T* array;
		/** the number of potentially used elements in array */
		int32_t array_size;
		/** if array must be freed */
		bool free_array;
		/** array's name */
		const char* name;
		/** array statistics */
		DECLARE_ARRAY_STATISTICS;

};
}
#endif /* _ARRAY_H_ */
