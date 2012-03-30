/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg, Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _ARRAY2_H_
#define _ARRAY2_H_

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/Array.h>

namespace shogun
{
template <class T> class CArray2;

/** @brief Template class Array2 implements a dense two dimensional array.
 *
 * Note that depending on compile options everything will be inlined, such that
 * this is as high performance 2d-array implementation \b without error checking.
 *
 * */
template <class T> class CArray2: public CArray<T>
{
	public:
		/** constructor
		 *
		 * @param dim1 dimension 1
		 * @param dim2 dimension 2
		 */
		CArray2(int32_t dim1=1, int32_t dim2=1)
		: CArray<T>(dim1*dim2), dim1_size(dim1), dim2_size(dim2)
		{
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param dim1 dimension 1
		 * @param dim2 dimension 2
		 * @param p_free_array if array must be freed
		 * @param p_copy_array if array must be copied
		 */
		CArray2(T* p_array, int32_t dim1, int32_t dim2, bool p_free_array=true, bool p_copy_array=false)
		: CArray<T>(p_array, dim1*dim2, p_free_array, p_copy_array),
			dim1_size(dim1), dim2_size(dim2)
		{
		}

		/** constructor
		 *
		 * @param p_array another array
		 * @param dim1 dimension 1
		 * @param dim2 dimension 2
		 */
		CArray2(const T* p_array, int32_t dim1, int32_t dim2)
		: CArray<T>(p_array, dim1*dim2), dim1_size(dim1), dim2_size(dim2)
		{
		}

		virtual ~CArray2() {}

		/** return total array size (including granularity buffer)
		 *
		 * @param dim1 dimension 1 will be stored here
		 * @param dim2 dimension 2 will be stored here
		 */
		inline void get_array_size(int32_t & dim1, int32_t & dim2)
		{
			dim1=dim1_size;
			dim2=dim2_size;
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

		/** zero array */
		inline void zero() { CArray<T>::zero(); }

		/** set array with a constant */
		inline void set_const(T const_elem)
		{
			CArray<T>::set_const(const_elem) ;
		}

		/** get the array
		 * call get_array just before messing with it DO NOT call any
		 * [],resize/delete functions after get_array(), the pointer may
		 * become invalid !
		 *
		 * @return the array
		 */
		inline T* get_array() { return CArray<T>::array; }

		/** set array's name
		 *
		 * @param p_name new name
		 */
		inline void set_array_name(const char * p_name)
		{
			CArray<T>::set_array_name(p_name);
		}

		/** set the array pointer and free previously allocated memory
		 *
		 * @param p_array another array
		 * @param dim1 dimension 1
		 * @param dim2 dimensino 2
		 * @param p_free_array if array must be freed
		 * @param copy_array if array must be copied
		 */
		inline void set_array(T* p_array, int32_t dim1, int32_t dim2, bool p_free_array=true, bool copy_array=false)
		{
			dim1_size=dim1;
			dim2_size=dim2;
			CArray<T>::set_array(p_array, dim1*dim2, p_free_array, copy_array);
		}

		/** resize array
		 *
		 * @param dim1 new dimension 1
		 * @param dim2 new dimension 2
		 * @return if resizing was successful
		 */
		inline bool resize_array(int32_t dim1, int32_t dim2)
		{
			dim1_size=dim1;
			dim2_size=dim2;
			return CArray<T>::resize_array(dim1*dim2);
		}

		/** get array element at index
		 *
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @return array element at index
		 */
		inline const T& get_element(int32_t idx1, int32_t idx2) const
		{
			ARRAY_ASSERT(idx1>=0 && idx1<dim1_size);
			ARRAY_ASSERT(idx2>=0 && idx2<dim2_size);
			return CArray<T>::get_element(idx1+dim1_size*idx2);
		}

		/** set array element at index 'index'
		 *
		 * @param p_element array element
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @return if setting was successful
		 */
		inline bool set_element(const T& p_element, int32_t idx1, int32_t idx2)
		{
			ARRAY_ASSERT(idx1>=0 && idx1<dim1_size);
			ARRAY_ASSERT(idx2>=0 && idx2<dim2_size);
			return CArray<T>::set_element(p_element, idx1+dim1_size*idx2);
		}

		/** get array element at index
		 *
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @return array element at index
		 */
		inline const T& element(int32_t idx1, int32_t idx2) const
		{
			return get_element(idx1,idx2);
		}

		/** get array element at index
		 *
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @return array element at index
		 */
		inline T& element(int32_t idx1, int32_t idx2)
		{
			ARRAY_ASSERT((idx1>=0 && idx1<dim1_size));
			ARRAY_ASSERT((idx2>=0 && idx2<dim2_size));
			return CArray<T>::element(idx1+dim1_size*idx2);
		}

		/** get element of given array at given index
		 *
		 * @param p_array another array
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @return element of given array at given index
		 */
		inline T& element(T* p_array, int32_t idx1, int32_t idx2)
		{
			ARRAY_ASSERT(CArray<T>::array==p_array);
			ARRAY_ASSERT(idx1>=0 && idx1<dim1_size);
			ARRAY_ASSERT(idx2>=0 && idx2<dim2_size);
			return p_array[idx1+dim1_size*idx2];
		}

		/** get element of given array at given index
		 *
		 * @param p_array another array
		 * @param idx1 index 1
		 * @param idx2 index 2
		 * @param p_dim1_size size of dimension 1
		 * @return element of given array at given index
		 */
		inline T& element(T* p_array, int32_t idx1, int32_t idx2, int32_t p_dim1_size)
		{
			ARRAY_ASSERT(CArray<T>::array==p_array);
			ARRAY_ASSERT(p_dim1_size==dim1_size);
			ARRAY_ASSERT(idx1>=0 && idx1<p_dim1_size);
			ARRAY_ASSERT(idx2>=0 && idx2<dim2_size);
			return p_array[idx1+p_dim1_size*idx2];
		}

		/** operator overload for array assignment
		 *
		 * @param orig original array
		 * @return new array
		 */
		CArray2<T>& operator=(CArray2<T>& orig)
		{
			CArray<T>::operator=(orig);
			dim1_size=orig.dim1_size;
			dim2_size=orig.dim2_size;
			return *this;
		}

		/** display array */
		void display_array() const
		{
			if (CArray<T>::get_name())
				CArray<T>::SG_PRINT( "2d-Array '%s' of size: %dx%d\n", CArray<T>::get_name(), dim1_size,dim2_size);
			else
				CArray<T>::SG_PRINT( "2d-Array of size: %dx%d\n",dim1_size,dim2_size);
			for (int32_t i=0; i<dim1_size; i++)
			{
				CArray<T>::SG_PRINT( "element(%d,:) = [ ",i);
				for (int32_t j=0; j<dim2_size; j++)
					CArray<T>::SG_PRINT( "%1.1f,", (float32_t) element(i,j));
				CArray<T>::SG_PRINT( " ]\n");
			}
		}

		/** display array size */
		void display_size() const
		{
			CArray<T>::SG_PRINT( "2d-Array of size: %dx%d\n",dim1_size,dim2_size);
		}

		/** @return object name */
		inline virtual const char* get_name() { return "Array2"; }

	protected:
		/** size of array's dimension 1 */
		int32_t dim1_size;
		/** size of array's dimension 2 */
		int32_t dim2_size;
};
}
#endif
