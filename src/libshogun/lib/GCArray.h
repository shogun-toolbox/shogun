/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg, Fabio De Bona
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __GCARRAY_H__
#define __GCARRAY_H__
template <class T> class CArray : public CSGObject
{
	public:
		CArray(int32_t sz) : CSGObject()
		{
			ASSERT(sz>0);
			array = new T[sz];
			size=sz;
		}

		virtual ~CArray()
		{
			for (int32_t i=0; i<size; i++)
				SG_UNREF(array[i]);
			delete[] array;
		}

		inline void operator[](const T* element, int32_t index)
		{
			ASSERT(index>=0);
			ASSERT(index<size);
			SG_UNREF(array[index]);
			array[index]=element;
			SG_REF(element);
			return element;
		}

		inline const T& operator[](int32_t index) const
		{
			ASSERT(index>=0);
			ASSERT(index<size);
			T* element=array[index];
			SG_REF(element); //???
			return element;
		}

	protected:
		T* array;
		int32_t size;
};
#endif //__GCARRAY_H__
