/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SET_H_
#define _SET_H_

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "base/SGObject.h"

/** @brief Template Set class 
 *
 * Set grows and shrinks dynamically, while elements can be accessed via index.
 * It is performance tuned for simple types like float etc. and for hi-level
 * objects only stores pointers. Note that objects are not automagically SG_REF'd/deleted.
 */
template <class T> class CSet : public CSGObject
{
	public:
		CSet()
		{
			array = new CDynamicArray<T>();
			SG_REF(array);
		}

		~CSet()
		{
			SG_UNREF(array);
		}

		inline void add(T e)
		{
			if (!contains(e))
				array->append_element(e);
		}

		inline void remove(T e)
		{
			int32_t idx=array->find_element(e);
			if (idx>=0)
				array->delete_element(idx);
		}

		inline bool contains(T e)
		{
			int32_t idx=array->find_element(e);
			return (idx!=-1);
		}

		/** get number of elements
		 *
		 * @return number of elements
		 */
		inline int32_t get_num_elements() const
		{
			return array->get_num_elements();
		}

		/** get set element at index
		 *
		 * (does NOT do bounds checking)
		 *
		 * @param index index
		 * @return array element at index
		 */
		inline T get_element(int32_t index) const
		{
			return array->get_element(index);
		}

		/** operator overload for set read only access
		 * use add() for write access 
		 *
		 * DOES NOT DO ANY BOUNDS CHECKING
		 *
		 * @param index index
		 * @return element at index
		 */
		inline T operator[](int32_t index) const
		{
			return array->get_element(index);
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "Set"; }

	protected:
		CDynamicArray<T>* array;
};

#endif //_SET_H_
