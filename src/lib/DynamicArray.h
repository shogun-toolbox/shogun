/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg, Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DYNARRAY_H_
#define _DYNARRAY_H_

#include "lib/common.h"
#include "lib/Array.h"

/** dynamic array, i.e. array that can be used like a list or an array.
it grows and shrinks dynamically, while elements can be accessed via index
performance tuned for simple types as float etc.
for hi-level objects only store pointers
*/
template <class T> class CDynamicArray;

template <class T> class CDynamicArray: public CArray<T>
{
public:

CDynamicArray(INT p_resize_granularity = 128, INT p_last_element_idx=-1)
	: CArray<T>(resize_granularity), resize_granularity(p_resize_granularity)
	{
		last_element_idx = p_last_element_idx;
	}
	
CDynamicArray(T* p_array, INT p_num_elements, INT p_array_size, bool p_free_array=true, bool p_copy_array=false)
	: CArray<T>(p_array, p_array_size, p_free_array, p_copy_array)
	{
		last_element_idx = p_num_elements ;
	}
	
CDynamicArray(const T* p_array, INT p_num_elements, INT p_array_size)
	: CArray<T>(p_array, p_array_size)
	{
		last_element_idx = p_num_elements ;
	}
	
	~CDynamicArray()
	{
		CIO::message(M_DEBUG, "destroying CDynamicArray array (last_element_idx=%i)\n", last_element_idx) ;
	}
	
	/// set the resize granularity and return what has been set (minimum is 128) 
	inline INT set_granularity(INT g)
	{
		g=CMath::max(g,128);
		this->resize_granularity = g;
		return g;
	}

	/// return index of element which is at the end of the array
	inline INT get_num_elements()
	{
		return last_element_idx+1;
	}
	
	///return array element at index
	inline const T& get_element(INT index) const
	{
		ARRAY_ASSERT((CArray<T>::array != NULL) && (index >= 0) && (index <= last_element_idx));
		return CArray<T>::get_element(index) ;
	}
	
	///set array element at index 'index' return false in case of trouble
	inline bool set_element(const T& element, INT index)
	{
		ARRAY_ASSERT((CArray<T>::array != NULL) && (index >= 0));
		if (index <= last_element_idx)
		{
			CArray<T>::set_element(element, index) ;
			return true;
		}
		else if (index < CArray<T>::num_elements)
		{
			CArray<T>::set_element(element, index) ;
			last_element_idx=index;
			return true;
		}
		else
		{
			if (resize_array(index))
				return CArray<T>::set_element(element, index);
			else
				return false;
		}
	}
	
	inline const T& element(INT idx1) const
	{
		return get_element(idx1) ;
	}

	inline T& element(INT index) 
	{
		ARRAY_ASSERT((CArray<T>::array != NULL) && (index >= 0));
		if (index <= last_element_idx)
		{
			return CArray<T>::element(index) ;
		}
		else if (index < CArray<T>::array_size)
		{
			last_element_idx=index;
			return CArray<T>::element(index) ;
		}
		else
		{
			resize_array(index) ;
			return CArray<T>::element(index);
		}
	}
	
	/*inline T& element(T* p_array, INT index) 
	{
		ARRAY_ASSERT((CArray<T>::array != NULL) && (index >= 0));
		ARRAY_ASSERT(CArray<T>::array == p_array) ;
		if (index <= last_element_idx)
		{
			return p_array[index] ;
		}
		else if (index < CArray<T>::num_elements)
		{
			last_element_idx=index;
			return p_array[index] ;
		}
		else
		{
			ASSERT(0) ;
			resize_array(index) ;
			return element(index);
		}
		}*/

	/// clear the array (with zeros)
	inline void clear_array()
	{
		if (last_element_idx >= 0)
			memset(CArray<T>::array, 0, (last_element_idx+1)*sizeof(T));
	}

	///set array element at index 'index' return false in case of trouble
	inline bool insert_element(const T& element, INT index)
	{
		if (append_element(get_element(last_element_idx)))
		{
			for (INT i=last_element_idx-1; i>index; i--)
			{
				CArray<T>::element(i) = CArray<T>::element(i-1) ;
			}
			CArray<T>::element(index)=element ;
		}
		else
			return false;
	}

	///set array element at index 'index' return false in case of trouble
	inline bool append_element(const T& new_element)
	{
		element(last_element_idx+1) = new_element ;
		return true ;
	}

	///delete array element at idx (does not call delete[] or the like)
	inline bool delete_element(INT idx)
	{
		if (idx>=0 && idx<=last_element_idx)
		{
			for (INT i=idx; i<last_element_idx; i++)
				CArray<T>::element(i)=CArray<T>::element(i+1) ;

			CArray<T>::element(last_element_idx)=0 ;
			last_element_idx--;

			if ( CArray<T>::num_elements - last_element_idx >= resize_granularity)
				resize_array(last_element_idx);
		}
		else
			return false;
	}
	
	///resize the array 
	bool resize_array(INT n)
	{
		INT n_orig = n ;
		
		// one cannot shrink below resize_granularity
		if (n<resize_granularity)
			n=resize_granularity;

		if (!CArray<T>::resize_array(n))
			return false ;
		
		//in case of shrinking we must adjust last element idx
		if (n_orig-1<last_element_idx)
			last_element_idx=n_orig-1;
		
		return true;
	}
	
	/// set the array pointer and free previously allocated memory
	inline void set_array(T* array, INT num_elements, INT p_array_size, bool free_array=true, bool copy_array=false)
	{
		CArray<T>::set_array(array, p_array_size, free_array, copy_array) ;
		this->last_element_idx=num_elements-1;
	}

	///// operator overload for array assignment
	CDynamicArray<T>& operator=(CDynamicArray<T>& orig)
	{
		CArray<T>::operator=(orig) ;
		resize_granularity = orig.resize_granularity;
		last_element_idx = orig.last_element_idx;

		return *this;
	}

protected:
	/// shrink/grow step size
	INT resize_granularity;

	/// the element in the array that has largest index
	INT last_element_idx;
};
#endif
