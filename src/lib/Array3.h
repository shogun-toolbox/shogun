/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg, Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _ARRAY3_H_
#define _ARRAY3_H_

#include "lib/common.h"
#include "base/SGObject.h"
#include "lib/Array.h"

template <class T> class CArray3;

template <class T> class CArray3: public CArray<T>
{
public:

CArray3()
	: CArray<T>(1), dim1_size(1), dim2_size(1), dim3_size(1)
	{
	}

CArray3(INT dim1, INT dim2, INT dim3)
	: CArray<T>(dim1*dim2*dim3), dim1_size(dim1), dim2_size(dim2), dim3_size(dim3)
	{
	}

CArray3(T* p_array, INT dim1, INT dim2, INT dim3, bool p_free_array=true, bool p_copy_array=false)
	: CArray<T>(p_array, dim1*dim2*dim3, p_free_array, p_copy_array),
		dim1_size(dim1), dim2_size(dim2), dim3_size(dim3)
	{
	}

	
CArray3(const T* p_array, INT dim1, INT dim2, INT dim3)
	: CArray<T>(p_array, dim1*dim2*dim3),
		dim1_size(dim1), dim2_size(dim2), dim3_size(dim3)
	{
	}
	
	~CArray3()
	{
	}
	
	inline void set_name(const char * p_name) 
	{
		CArray<T>::set_name(p_name) ;
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
	inline void set_array(T* p_array, INT dim1, INT dim2, INT dim3, bool p_free_array, bool copy_array=false)
	{
		dim1_size=dim1 ;
		dim2_size=dim2 ;
		dim3_size=dim3 ;
		CArray<T>::set_array(p_array, dim1*dim2*dim3, p_free_array, copy_array) ;
	}

	inline bool resize_array(INT dim1, INT dim2, INT dim3)
		{
			dim1_size=dim1 ;
			dim2_size=dim2 ;
			dim3_size=dim3 ;
			return CArray<T>::resize_array(dim1*dim2*dim3) ;
		}

	///return array element at index
	inline T get_element(INT idx1, INT idx2, INT idx3) const
	{
		ASSERT(idx1>=0 && idx1<dim1_size) ;		
		ASSERT(idx2>=0 && idx2<dim2_size) ;		
		ASSERT(idx3>=0 && idx3<dim3_size) ;		
		return CArray<T>::get_element(idx1+dim1_size*(idx2+dim2_size*idx3)) ;
	}

	///set array element at index 'index' return false in case of trouble
	inline bool set_element(T p_element, INT idx1, INT idx2, INT idx3)
	{
		ASSERT(idx1>=0 && idx1<dim1_size) ;		
		ASSERT(idx2>=0 && idx2<dim2_size) ;		
		ASSERT(idx3>=0 && idx3<dim3_size) ;		
		return CArray<T>::set_element(p_element, idx1+dim1_size*(idx2+dim2_size*idx3)) ;
	}

	inline const T& element(INT idx1, INT idx2, INT idx3) const
	{
		ASSERT(idx1>=0 && idx1<dim1_size) ;		
		ASSERT(idx2>=0 && idx2<dim2_size) ;		
		ASSERT(idx3>=0 && idx3<dim3_size) ;		
		return CArray<T>::element(idx1+dim1_size*(idx2+dim2_size*idx3)) ;
	}

	inline T& element(INT idx1, INT idx2, INT idx3) 
	{
		ASSERT(idx1>=0 && idx1<dim1_size) ;		
		ASSERT(idx2>=0 && idx2<dim2_size) ;		
		ASSERT(idx3>=0 && idx3<dim3_size) ;		
		return CArray<T>::element(idx1+dim1_size*(idx2+dim2_size*idx3)) ;
	}

	inline T& element(T* p_array, INT idx1, INT idx2, INT idx3) 
	{
		ASSERT(p_array==CArray<T>::array) ;		
		ASSERT(idx1>=0 && idx1<dim1_size) ;		
		ASSERT(idx2>=0 && idx2<dim2_size) ;		
		ASSERT(idx3>=0 && idx3<dim3_size) ;		
		return p_array[idx1+dim1_size*(idx2+dim2_size*idx3)] ;
	}
	
	inline T& element(T* p_array, INT idx1, INT idx2, INT idx3, INT p_dim1_size, INT p_dim2_size) 
	{
		ASSERT(p_array==CArray<T>::array) ;		
		ASSERT(p_dim1_size==dim1_size) ;		
		ASSERT(p_dim2_size==dim2_size) ;		
		ASSERT(idx1>=0 && idx1<p_dim1_size) ;		
		ASSERT(idx2>=0 && idx2<p_dim2_size) ;		
		ASSERT(idx3>=0 && idx3<dim3_size) ;		
		return p_array[idx1+p_dim1_size*(idx2+p_dim2_size*idx3)] ;
	}
	
	///// operator overload for array assignment
	CArray3<T>& operator=(const CArray3<T>& orig)
	{
		CArray<T>::operator=(orig) ;
		dim1_size=orig.dim1_size ;
		dim2_size=orig.dim2_size ;
		dim3_size=orig.dim3_size ;
		return *this;
	}

	void display_size() const
	{
		CArray<T>::SG_PRINT( "3d-Array of size: %dx%dx%d\n",dim1_size, dim2_size, dim3_size);
	}

	void display_array() const
	{
		if (CArray<T>::get_name())
			CArray<T>::SG_PRINT( "3d-Array '%s' of size: %dx%dx%d\n", CArray<T>::get_name(), dim1_size, dim2_size, dim3_size);
		else
			CArray<T>::SG_PRINT( "2d-Array of size: %dx%dx%d\n",dim1_size, dim2_size, dim3_size);
		for (INT k=0; k<dim3_size; k++)
			for (INT i=0; i<dim1_size; i++)
			{
				CArray<T>::SG_PRINT( "element(%d,:,%d) = [ ",i, k);
				for (INT j=0; j<dim2_size; j++)
					CArray<T>::SG_PRINT( "%1.1f,", (float)element(i,j,k));
				CArray<T>::SG_PRINT( " ]\n");
			}
	}

protected:
	/// the number of potentially used elements in array
	INT dim1_size;
	INT dim2_size;
	INT dim3_size;
};
#endif
