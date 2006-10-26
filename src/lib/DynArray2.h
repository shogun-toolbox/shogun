/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DYNARRAY2_H_
#define _DYNARRAY2_H_

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/DynArray.h"

/** dynamic array, i.e. array that can be used like a list or an array.
it grows and shrinks dynamically, while elements can be accessed via index
performance tuned for simple types as float etc.
for hi-level objects only store pointers
*/
template <class T> class CDynamicArray2;

template <class T> class CDynamicArray2: CDynamicArray<T>
{
public:
	CDynamicArray2(INT dim1, INT dim2)
		: CDynamicArray<T>(dim1*dim2), dim1_size(dim1), dim2_size(dim2)
	{
		this->resize_granularity = dim1 ;
		last_element_idx=dim1*dim2-1 ;
	}

	CDynamicArray2(T* p_array, INT dim1, INT dim2, bool p_free_array=true)
		: CDynamicArray<T>(p_array, dim1*dim2, dim1*dim2, p_free_array),
		dim1_size(dim1), dim2_size(dim2)
		{
		}

	~CDynamicArray2()
	{
	}

	/// return total array size (including granularity buffer)
	inline void get_array_size(INT & dim1, INT & dim2)
	{
		dim1=dim1_size ;
		dim2=dim2_size ;
	}

	/// set the array pointer and free previously allocated memory
	inline void set_array(T* array, INT dim1, INT dim2, bool free_array=true, bool copy_array=false)
		{
			dim1_size=dim1 ;
			dim2_size=dim2 ;
			CDynamicArray<T>::
set_array(array, dim1*dim2, dim1*dim2, free_array, copy_array) ;
		}

	inline bool resize_array(INT dim1, INT dim2)
		{
			dim1_size=dim1 ;
			dim2_size=dim2 ;
			if (CDynamicArray<T>::resize_array(dim1*dim2))
			{
				last_element_idx=dim1*dim2-1 ;
				return true ;
			}
			else 
				return false ;
		}

	///return array element at index
	inline const T& get_element(INT idx1, INT idx2) const
	{
		ASSERT(idx1>=0 && idx1<dim1_size) ;		
		ASSERT(idx2>=0 && idx2<dim2_size) ;		
		return get_element(idx1+dim1_size*idx2) ;
	}

	///set array element at index 'index' return false in case of trouble
	inline bool set_element(const T& element, INT idx1, INT idx2)
	{
		ASSERT(idx1>=0 && idx1<dim1_size) ;		
		ASSERT(idx2>=0 && idx2<dim2_size) ;		
		return set_element(element, idx1+dim1_size*idx2) ;
	}
	
	inline const T& element(INT idx1, INT idx2) const
	{
		return element(idx1,idx2) ;
	}

	inline T& element(INT idx1, INT idx2) 
	{
		ASSERT(idx1>=0 && idx1<dim1_size) ;		
		ASSERT(idx2>=0 && idx2<dim2_size) ;		
		return element(idx1,idx2) ;
	}

	///// operator overload for array assignment
	CDynamicArray<T>& operator=(CDynamicArray<T>& orig)
	{
		CDynamicArray<T>::operator=(orig) ;
		dim1_size=orig.dim1_size ;
		dim2_size=orig.dim2_size ;
		return *this;
	}

protected:
	/// the number of potentially used elements in array
	INT dim1_size;
	INT dim2_size;
};
#endif
