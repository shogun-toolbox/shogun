/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg, Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _ARRAY2_H_
#define _ARRAY2_H_

#include "lib/common.h"
#include "lib/Array.h"

template <class T> class CArray2;

template <class T> class CArray2: CArray<T>
{
public:
CArray2(INT dim1, INT dim2)
	: CArray<T>(dim1*dim2), dim1_size(dim1), dim2_size(dim2)
	{
	}

CArray2(T* p_array, INT dim1, INT dim2, bool p_free_array=true, bool p_copy_array=false)
	: CArray<T>(p_array, dim1*dim2, p_free_array, p_copy_array),
		dim1_size(dim1), dim2_size(dim2)
		{
		}

CArray2(const T* p_array, INT dim1, INT dim2)
	: CArray<T>(p_array, dim1*dim2),
		dim1_size(dim1), dim2_size(dim2)
		{
		}

	~CArray2()
	{
#ifdef ARRAY_STATISTICS
		CIO::message(M_DEBUG, "destroying CArray2 array of size %i x %i\n", dim1_size, dim2_size) ;
#endif
	}

#ifdef ARRAY_STATISTICS
	inline void set_name(const char * p_name) 
	{
		CArray<T>::set_name(p_name) ;
	}
#endif

	/// return total array size (including granularity buffer)
	inline void get_array_size(INT & dim1, INT & dim2)
	{
		dim1=dim1_size ;
		dim2=dim2_size ;
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

	inline void zero()
	{
		CArray<T>::zero() ;
	}

	/// get the array
	/// call get_array just before messing with it DO NOT call any [],resize/delete functions after get_array(), the pointer may become invalid !
	inline T* get_array()
	{
		return CArray<T>::array;
	}

	/// set the array pointer and free previously allocated memory
	inline void set_array(T* array, INT dim1, INT dim2, bool free_array=true, bool copy_array=false)
		{
			dim1_size=dim1 ;
			dim2_size=dim2 ;
			CArray<T>::set_array(array, dim1*dim2, free_array, copy_array) ;
		}

	inline bool resize_array(INT dim1, INT dim2)
		{
			dim1_size=dim1 ;
			dim2_size=dim2 ;
			return CArray<T>::resize_array(dim1*dim2) ;
		}

	///return array element at index
	inline const T& get_element(INT idx1, INT idx2) const
	{
		ARRAY_ASSERT(idx1>=0 && idx1<dim1_size) ;		
		ARRAY_ASSERT(idx2>=0 && idx2<dim2_size) ;		
		return CArray<T>::get_element(idx1+dim1_size*idx2) ;
	}

	///set array element at index 'index' return false in case of trouble
	inline bool set_element(const T& element, INT idx1, INT idx2)
	{
		ARRAY_ASSERT(idx1>=0 && idx1<dim1_size) ;		
		ARRAY_ASSERT(idx2>=0 && idx2<dim2_size) ;		
		return CArray<T>::set_element(element, idx1+dim1_size*idx2) ;
	}
	
	inline const T& element(INT idx1, INT idx2) const
	{
		return get_element(idx1,idx2) ;
	}
	
	inline T& element(INT idx1, INT idx2) 
	{
		ARRAY_ASSERT(idx1>=0 && idx1<dim1_size) ;		
		ARRAY_ASSERT(idx2>=0 && idx2<dim2_size) ;		
		return CArray<T>::element(idx1+dim1_size*idx2) ;
	}

	inline T& element(T* p_array, INT idx1, INT idx2) 
	{
		ARRAY_ASSERT(CArray<T>::array==p_array) ;
		ARRAY_ASSERT(idx1>=0 && idx1<dim1_size) ;		
		ARRAY_ASSERT(idx2>=0 && idx2<dim2_size) ;		
#ifdef ARRAY_STATISTICS
		CArray<T>::stat_array_element++ ;
#endif
		return p_array[idx1+dim1_size*idx2] ;
	}

	inline T& element(T* p_array, INT idx1, INT idx2, INT p_dim1_size) 
	{
		ARRAY_ASSERT(CArray<T>::array==p_array) ;
		ARRAY_ASSERT(p_dim1_size==dim1_size) ;
		ARRAY_ASSERT(idx1>=0 && idx1<p_dim1_size) ;		
		ARRAY_ASSERT(idx2>=0 && idx2<dim2_size) ;		
#ifdef ARRAY_STATISTICS
		CArray<T>::stat_array_element++ ;
#endif
		return p_array[idx1+p_dim1_size*idx2] ;
	}

	///// operator overload for array assignment
	CArray<T>& operator=(CArray<T>& orig)
	{
		CArray<T>::operator=(orig) ;
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
