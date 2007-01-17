	/*
	 * This program is free software; you can redistribute it and/or modify
	 * it under the terms of the GNU General Public License as published by
	 * the Free Software Foundation; either version 2 of the License, or
	 * (at your option) any later version.
	 *
	 * Written (W) 1999-2007 Soeren Sonnenburg, Gunnar Raetsch
	 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
	 */

	#ifndef _ARRAY_H_
	#define _ARRAY_H_

	#include <assert.h>

	//#define ARRAY_STATISTICS
	#ifdef ASSERT
	#undef ASSERT
	#endif
	#define ASSERT(x)

	#include "lib/common.h"

	template <class T> class CArray;

	template <class T> class CArray
	{
	public:
	CArray(INT initial_size = 1)
		: free_array(true), name(NULL)
	#ifdef ARRAY_STATISTICS
			, stat_const_element(0), stat_element(0), stat_set_element(0), stat_get_element(0), stat_operator(0), stat_const_operator(0), stat_set_array(0), stat_get_array(0), stat_resize_array(0), stat_array_element(0)
	#endif
		{
			array_size = initial_size;
			array = (T*) calloc(array_size, sizeof(T));
			ASSERT(array);
		}
		
	CArray(T* p_array, INT p_array_size, bool p_free_array=true, bool p_copy_array=false)
		: array(NULL), free_array(false), name(NULL)
	#ifdef ARRAY_STATISTICS
			, stat_const_element(0), stat_element(0), stat_set_element(0), stat_get_element(0), stat_operator(0), stat_const_operator(0), stat_set_array(0), stat_get_array(0), stat_resize_array(0), stat_array_element(0)
	#endif
		{
			set_array(p_array, p_array_size, p_free_array, p_copy_array) ;
		}
		
	CArray(const T* p_array, INT p_array_size)
		: array(NULL), free_array(false), name(NULL)
	#ifdef ARRAY_STATISTICS
			, stat_const_element(0), stat_element(0), stat_set_element(0), stat_get_element(0), stat_operator(0), stat_const_operator(0), stat_set_array(0), stat_get_array(0), stat_resize_array(0), stat_array_element(0)
	#endif
		{
			set_array(p_array, p_array_size) ;
		}
		
		~CArray()
		{
	#ifdef ARRAY_STATISTICS
			if (!name)
				name="unnamed" ;
			CIO::message(M_DEBUG, "destroying CArray array '%s' of size %i\n", name, array_size) ;
			CIO::message(M_DEBUG, "access statistics:\nconst element    %i\nelement    %i\nset_element    %i\nget_element    %i\nstat_operator[]    %i\nconst_operator[]    %i\nset_array    %i\nget_array    %i\nresize_array    %i\narray_element    %i\n", stat_const_element, stat_element, stat_set_element, stat_get_element, stat_operator, stat_const_operator, stat_set_array, stat_get_array, stat_resize_array, stat_array_element) ;
	#endif
			if (free_array)
				free(array);
		}

		inline void set_name(const char * p_name) 
		{
			name = p_name ;
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
	#ifdef ARRAY_STATISTICS
			((CArray<T>*)this)->stat_get_element++ ;
	#endif
			return array[index];
		}
		
		///set array element at index 'index' return false in case of trouble
		inline bool set_element(const T& p_element, INT index)
		{
			ASSERT((array != NULL) && (index >= 0) && (index < array_size));
	#ifdef ARRAY_STATISTICS
			((CArray<T>*)this)->stat_set_element++ ;
	#endif
			array[index]=p_element;
			return true;
		}
		
		inline const T& element(INT idx1) const
		{
	#ifdef ARRAY_STATISTICS
			// hack to get rid of the const
			((CArray<T>*)this)->stat_const_element++ ;
	#endif
			return get_element(idx1) ;
		}

		inline T& element(INT index) 
		{
		ASSERT(array != NULL);
		ASSERT(index >= 0);
		ASSERT(index < array_size);
#ifdef ARRAY_STATISTICS
		((CArray<T>*)this)->stat_element++ ;
#endif
		return array[index] ;
	}

	inline T& element(T* p_array, INT index) 
	{
		ASSERT((array != NULL) && (index >= 0) && (index < array_size));
		ASSERT(array == p_array) ;
#ifdef ARRAY_STATISTICS
		((CArray<T>*)this)->stat_array_element++ ;
#endif
		return p_array[index] ;
	}
	
	///resize the array 
	bool resize_array(INT n)
	{
#ifdef ARRAY_STATISTICS
		((CArray<T>*)this)->stat_resize_array++ ;
#endif
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
#ifdef ARRAY_STATISTICS
		((CArray<T>*)this)->stat_get_array++ ;
#endif
		return array;
	}

	/// set the array pointer and free previously allocated memory
	inline void set_array(T* p_array, INT p_array_size, bool p_free_array=true, bool copy_array=false)
	{
#ifdef ARRAY_STATISTICS
		((CArray<T>*)this)->stat_set_array++ ;
#endif
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
#ifdef ARRAY_STATISTICS
		((CArray<T>*)this)->stat_set_array++ ;
#endif
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
#ifdef ARRAY_STATISTICS
		((CArray<T>*)this)->stat_const_operator++ ;
#endif
		return array[index];
	}
	
	inline T& operator[](INT index) 
	{
#ifdef ARRAY_STATISTICS
		((CArray<T>*)this)->stat_operator++ ;
#endif
		return element(index);
	}
	
	///// operator overload for array assignment
	CArray<T>& operator=(CArray<T>& orig)
	{
		memcpy(array, orig.array, sizeof(T)*orig.array_size);
		array_size=orig.array_size;

		return *this;
	}

	void display_array() const
	{
		if (!name)
			CIO::message(M_MESSAGEONLY, "Array of size: %d\n", array_size);
		else
			CIO::message(M_MESSAGEONLY, "Array '%s' of size: %d\n", name, array_size);

		for (INT i=0; i<array_size; i++)
			CIO::message(M_MESSAGEONLY, "%d,", array[i]);
		CIO::message(M_MESSAGEONLY, "\n");
	}

	void display_size() const
	{
		if (!name)
			CIO::message(M_MESSAGEONLY, "Array of size: %d\n", array_size);
		else
			CIO::message(M_MESSAGEONLY, "Array '%s' of size: %d\n", name, array_size);
	}
	
protected:

	/// memory for dynamic array
	T* array;

	/// the number of potentially used elements in array
	INT array_size;

	/// 
	bool free_array ;

	const char * name ;
#ifdef ARRAY_STATISTICS
	INT stat_const_element, stat_element, stat_set_element, stat_get_element, stat_operator, stat_const_operator, stat_set_array, stat_get_array, stat_resize_array, stat_array_element;
#endif
};
#endif
