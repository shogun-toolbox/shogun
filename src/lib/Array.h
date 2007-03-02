/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg, Gunnar Raetsch, Andre Noll
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
#include "base/SGObject.h"

#ifdef ARRAY_STATISTICS
struct array_statistics {
	INT const_element;
	INT element;
	INT set_element;
	INT get_element;
	INT operator_overload;
	INT const_operator_overload;
	INT set_array;
	INT get_array;
	INT resize_array;
	INT array_element;
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

template <class T> class CArray : public CSGObject
{
public:
	CArray(INT initial_size = 1)
	: CSGObject(), free_array(true), name(NULL)
	{
		INIT_ARRAY_STATISTICS;
		array_size = initial_size;
		array = (T*) calloc(array_size, sizeof(T));
		ASSERT(array);
	}

	CArray(T* p_array, INT p_array_size, bool p_free_array=true,
			bool p_copy_array=false)
	: CSGObject(), array(NULL), free_array(false), name(NULL)
	{
		INIT_ARRAY_STATISTICS;
		set_array(p_array, p_array_size, p_free_array, p_copy_array);
	}

	CArray(const T* p_array, INT p_array_size)
	: CSGObject(), array(NULL), free_array(false), name(NULL)
	{
		INIT_ARRAY_STATISTICS;
		set_array(p_array, p_array_size);
	}

	~CArray()
	{
		SG_DEBUG( "destroying CArray array '%s' of size %i\n",
			name? name : "unnamed", array_size);
		PRINT_ARRAY_STATISTICS;
		if (free_array)
			free(array);
	}

	inline const char* get_name() const
	{
		return name;
	}

	inline void set_name(const char * p_name)
	{
		name = p_name;
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
			array[i]=0;
	}

	/// return array element at index
	inline const T& get_element(INT index) const
	{
		ASSERT((array != NULL) && (index >= 0) && (index < array_size));
		INCREMENT_ARRAY_STATISTICS_VALUE(get_element);
		return array[index];
	}

	/// set array element at index 'index' return false in case of trouble
	inline bool set_element(const T& p_element, INT index)
	{
		ASSERT((array != NULL) && (index >= 0) && (index < array_size));
		INCREMENT_ARRAY_STATISTICS_VALUE(set_element);
		array[index]=p_element;
		return true;
	}

	inline const T& element(INT idx1) const
	{
		INCREMENT_ARRAY_STATISTICS_VALUE(const_element);
		return get_element(idx1);
	}

	inline T& element(INT index)
	{
		ASSERT(array != NULL);
		ASSERT(index >= 0);
		ASSERT(index < array_size);
		INCREMENT_ARRAY_STATISTICS_VALUE(element);
		return array[index];
	}

	inline T& element(T* p_array, INT index)
	{
		ASSERT((array != NULL) && (index >= 0) && (index < array_size));
		ASSERT(array == p_array);
		INCREMENT_ARRAY_STATISTICS_VALUE(array_element);
		return p_array[index];
	}

	bool resize_array(INT n)
	{
		INCREMENT_ARRAY_STATISTICS_VALUE(resize_array);
		ASSERT(free_array);

		T* p= (T*) realloc(array, sizeof(T)*n);
		if (!p)
			return false;
		array=p;
		if (n > array_size)
			memset(&array[array_size], 0, (n-array_size)*sizeof(T));
		array_size=n;
		return true;
	}

	/// call get_array just before messing with it DO NOT call any
	///  [],resize/delete functions after get_array(), the pointer
	///  may become invalid!
	inline T* get_array()
	{
		INCREMENT_ARRAY_STATISTICS_VALUE(get_array);
		return array;
	}

	/// set the array pointer and free previously allocated memory
	inline void set_array(T* p_array, INT p_array_size, bool p_free_array=true,
			bool copy_array=false)
	{
		INCREMENT_ARRAY_STATISTICS_VALUE(set_array);
		if (this->free_array)
			free(this->array);
		if (copy_array)
		{
			this->array=(T*)malloc(p_array_size*sizeof(T));
			memcpy(this->array, p_array, p_array_size*sizeof(T));
		}
		else
			this->array=p_array;
		this->array_size=p_array_size;
		this->free_array=p_free_array;
	}

	/// set the array pointer and free previously allocated memory
	inline void set_array(const T* p_array, INT p_array_size)
	{
		INCREMENT_ARRAY_STATISTICS_VALUE(set_array);
		free(this->array);
		this->array=(T*)malloc(p_array_size*sizeof(T));
		memcpy(this->array, p_array, p_array_size*sizeof(T));
		this->array_size=p_array_size;
		this->free_array=true;
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
		INCREMENT_ARRAY_STATISTICS_VALUE(const_operator_overload);
		return array[index];
	}

	inline T& operator[](INT index)
	{
		INCREMENT_ARRAY_STATISTICS_VALUE(operator_overload);
		return element(index);
	}

	///// operator overload for array assignment
	CArray<T>& operator=(const CArray<T>& orig)
	{
		memcpy(array, orig.array, sizeof(T)*orig.array_size);
		array_size=orig.array_size;

		return *this;
	}

	void display_size() const
	{
		SG_PRINT( "Array '%s' of size: %d\n", name? name : "unnamed",
			array_size);
	}

	void display_array() const
	{
		display_size();
		for (INT i=0; i<array_size; i++)
			SG_PRINT("%1.1f,", (float)array[i]);
		SG_PRINT("\n");
	}

protected:
	/// memory for dynamic array
	T* array;
	/// the number of potentially used elements in array
	INT array_size;
	///
	bool free_array;
	const char *name;
	DECLARE_ARRAY_STATISTICS;

};
#endif /* _ARRAY_H_ */
