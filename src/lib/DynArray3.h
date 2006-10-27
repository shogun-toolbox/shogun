/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DYNARRAY3_H_
#define _DYNARRAY3_H_

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/DynArray.h"

/** dynamic array, i.e. array that can be used like a list or an array.
it grows and shrinks dynamically, while elements can be accessed via index
performance tuned for simple types as float etc.
for hi-level objects only store pointers
*/
template <class T> class CDynamicArray3;

template <class T> class CDynamicArray3: CDynamicArray<T>
{
public:
	CDynamicArray3(INT dim1, INT dim2, INT dim3)
		: CDynamicArray<T>(dim1*dim2), dim1_size(dim1), dim2_size(dim2), dim3_size(dim3)
	{
		this->resize_granularity = dim1 ;
	}

CDynamicArray3(T* p_array, INT dim1, INT dim2, INT dim3, bool p_free_array=true, bool p_copy_array=false)
	: CDynamicArray<T>(p_array, dim1*dim2*dim3, dim1*dim2*dim3, p_free_array, p_copy_array),
		dim1_size(dim1), dim2_size(dim2), dim3_size(dim3)
		{
		}

CDynamicArray3(const T* p_array, INT dim1, INT dim2, INT dim3)
	: CDynamicArray<T>(p_array, dim1*dim2*dim3, dim1*dim2*dim3),
		dim1_size(dim1), dim2_size(dim2), dim3_size(dim3)
		{
		}

	virtual ~CDynamicArray3()
	{
	}

	/// return total array size (including granularity buffer)
	inline void get_array_size(INT & dim1, INT & dim2, INT & dim3)
	{
		dim1=dim1_size ;
		dim2=dim2_size ;
		dim3=dim3_size ;
	}
	/// return dimension 1
	inline INT get_dim1()
	{
		return dim1_size ;
	}

	/// return dimension 2
	inline INT get_dim2()
	{
		return dim2_size ;
	}

	/// return dimension 3
	inline INT get_dim3()
	{
		return dim3_size ;
	}

	/// get the array
	/// call get_array just before messing with it DO NOT call any [],resize/delete functions after get_array(), the pointer may become invalid !
	inline T* get_array()
	{
		return CDynamicArray<T>::array;
	}

	/// set the array pointer and free previously allocated memory
	inline void set_array(T* array, INT dim1, INT dim2, INT dim3, bool free_array, bool copy_array=false)
	{
		dim1_size=dim1 ;
		dim2_size=dim2 ;
		dim3_size=dim3 ;
		CDynamicArray<T>::set_array(array, dim1*dim2*dim3, dim1*dim2*dim3, free_array, copy_array) ;
	}

	inline bool resize_array(INT dim1, INT dim2, INT dim3)
		{
			dim1_size=dim1 ;
			dim2_size=dim2 ;
			dim3_size=dim3 ;
			if (CDynamicArray<T>::resize_array(dim1*dim2*dim3))
			{
				CDynamicArray<T>::last_element_idx=dim1*dim2*dim3-1 ;
				return true ;
			}
			else 
				return false ;
		}

	///return array element at index
	inline T get_element(INT idx1, INT idx2, INT idx3) const
	{
		ASSERT(idx1>=0 && idx1<dim1_size) ;		
		ASSERT(idx2>=0 && idx2<dim2_size) ;		
		ASSERT(idx3>=0 && idx3<dim3_size) ;		
		return CDynamicArray<T>::get_element(idx1+dim1_size*(idx2+dim2_size*idx3)) ;
	}

	///set array element at index 'index' return false in case of trouble
	inline bool set_element(T element, INT idx1, INT idx2, INT idx3)
	{
		ASSERT(idx1>=0 && idx1<dim1_size) ;		
		ASSERT(idx2>=0 && idx2<dim2_size) ;		
		ASSERT(idx3>=0 && idx3<dim3_size) ;		
		return CDynamicArray<T>::set_element(element, idx1+dim1_size*(idx2+dim2_size*idx3)) ;
	}

	inline const T& element(INT idx1, INT idx2, INT idx3) const
	{
		return get_element(idx1,idx2,idx3) ;
	}

	inline T& element(INT idx1, INT idx2, INT idx3) 
	{
		return CDynamicArray<T>::element(idx1+dim1_size*(idx2+dim2_size*idx3)) ;
	}
	
	///// operator overload for array assignment
	CDynamicArray<T>& operator=(CDynamicArray<T>& orig)
	{
		CDynamicArray<T>::operator=(orig) ;
		dim1_size=orig.dim1_size ;
		dim2_size=orig.dim2_size ;
		dim3_size=orig.dim3_size ;
		return *this;
	}

protected:
	/// the number of potentially used elements in array
	INT dim1_size;
	INT dim2_size;
	INT dim3_size;
};
#endif
