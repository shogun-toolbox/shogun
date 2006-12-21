/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg, Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _ARRAY_H_
#define _ARRAY_H_

#include "lib/common.h"
#include "lib/io.h"

template <class T> class CArray;

template <class T> class CArray
{
public:
CArray(INT initial_size = 1)
	: free_array(true) 
	{
		array_size = initial_size;
		array = (T*) calloc(array_size, sizeof(T));
		ASSERT(array);
	}
	
CArray(T* p_array, INT p_array_size, bool p_free_array=true, bool p_copy_array=false)
	: array(NULL), free_array(false)
	{
		set_array(p_array, p_array_size, p_free_array, p_copy_array) ;
	}
	
CArray(const T* p_array, INT p_array_size)
	: array(NULL), free_array(false)
	{
		set_array(p_array, p_array_size) ;
	}
	
	~CArray()
	{
		if (free_array)
			free(array);
	}

	/// return total array size (including granularity buffer)
	inline INT get_array_size() const
	{
		return array_size;
	}

	/// return total array size (including granularity buffer)
	inline INT get_dim1()
	{
		return array_size;
	}

	inline void zero()
	{
		for (INT i=0; i< array_size; i++)
			array[i]=0 ;
	}
	
	///return array element at index
	inline const T& get_element(INT index) const
	{
		ASSERT((array != NULL) && (index >= 0) && (index < array_size));
		return array[index];
	}
	
	///set array element at index 'index' return false in case of trouble
	inline bool set_element(const T& p_element, INT index)
	{
		ASSERT((array != NULL) && (index >= 0) && (index < array_size));
		array[index]=p_element;
		return true;
	}
	
	inline const T& element(INT idx1) const
	{
		return get_element(idx1) ;
	}

	inline T& element(INT index) 
	{
		ASSERT((array != NULL) && (index >= 0) && (index < array_size));
		return array[index] ;
	}

	inline T& element(T* p_array, INT index) 
	{
		ASSERT((array != NULL) && (index >= 0) && (index < array_size));
		ASSERT(array == p_array) ;
		return p_array[index] ;
	}
	
	///resize the array 
	bool resize_array(INT n)
	{
		ASSERT(free_array) ;
		
		T* p= (T*) realloc(array, sizeof(T)*n);
		if (p)
		{
			array=p;
			if (n > array_size)
				memset(&array[array_size], 0, (n-array_size)*sizeof(T));

			array_size=n ;
			return true;
		}
		else
			return false;
	}

	/// get the array
	/// call get_array just before messing with it DO NOT call any [],resize/delete functions after get_array(), the pointer may become invalid !
	inline T* get_array()
	{
		return array;
	}

	/// set the array pointer and free previously allocated memory
	inline void set_array(T* p_array, INT p_array_size, bool p_free_array=true, bool copy_array=false)
	{
		if (this->free_array)
			free(this->array);
		if (copy_array)
		{
			this->array=(T*)malloc(p_array_size*sizeof(T)) ;
			memcpy(this->array, p_array, p_array_size*sizeof(T)) ;
		}
		else
			this->array=p_array;
		this->array_size=p_array_size;
		this->free_array=p_free_array ;
	}

	/// set the array pointer and free previously allocated memory
	inline void set_array(const T* p_array, INT p_array_size)
	{
		if (this->free_array)
			free(this->array);
		this->array=(T*)malloc(p_array_size*sizeof(T)) ;
		memcpy(this->array, p_array, p_array_size*sizeof(T)) ;
		this->array_size=p_array_size;
		this->free_array=true ;
	}

	/// clear the array (with zeros)
	inline void clear_array()
	{
		memset(array, 0, array_size*sizeof(T));
	}


	/// operator overload for array read only access
	/// use set_element() for write access (will also make the array dynamically grow)
	///
	/// DOES NOT DO ANY BOUNDS CHECKING
	inline const T& operator[](INT index) const
	{
		return array[index];
	}
	
	inline T& operator[](INT index) 
	{
		return element(index);
	}
	
	///// operator overload for array assignment
	CArray<T>& operator=(CArray<T>& orig)
	{
		memcpy(array, orig.array, sizeof(T)*orig.array_size);
		array_size=orig.array_size;

		return *this;
	}

protected:
	/// memory for dynamic array
	T* array;

	/// the number of potentially used elements in array
	INT array_size;

	/// 
	bool free_array ;
};
#endif
