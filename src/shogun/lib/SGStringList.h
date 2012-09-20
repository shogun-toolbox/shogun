/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Written (W) 2010,2012 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 * Copyright (C) 2012 Soeren Sonnenburg
 */
#ifndef __SGSTRINGLIST_H__
#define __SGSTRINGLIST_H__

#include <shogun/lib/config.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGString.h>
#include <shogun/lib/SGReferencedData.h>

namespace shogun
{
/** @brief template class SGStringList */
template <class T> struct SGStringList : public SGReferencedData
{
public:
	/** default constructor */
	SGStringList() : SGReferencedData()
	{
		init_data();
	}

	/** constructor for setting params */
	SGStringList(SGString<T>* s, index_t num_s, index_t max_length,
			bool ref_counting=true) : 
		SGReferencedData(ref_counting), num_strings(num_s),
		max_string_length(max_length), strings(s)
	{
	}

	/** constructor to create new string list in memory */
	SGStringList(index_t num_s, index_t max_length, bool ref_counting=true) : 
		SGReferencedData(ref_counting),
		num_strings(num_s), max_string_length(max_length)
	{
		strings = SG_MALLOC(SGString<T>, num_strings);
		for (int32_t i=0; i<num_strings; i++)
			new (&strings[i]) SGString<T>();
	}

	/** copy constructor */
	SGStringList(const SGStringList &orig) :
		SGReferencedData(orig)
	{
		copy_data(orig);
	}

	/** destructor */
	virtual ~SGStringList()
	{
		unref();
	}

protected:

	/** copy data */
	virtual void copy_data(const SGReferencedData &orig)
	{
		strings = ((SGStringList*)(&orig))->strings;
		num_strings = ((SGStringList*)(&orig))->num_strings;
		max_string_length = ((SGStringList*)(&orig))->max_string_length;
	}

	/** init data */
	virtual void init_data()
	{
		strings = NULL;
		num_strings = 0;
		max_string_length = 0;
	}

	/** free data */
	virtual void free_data()
	{
		for (int32_t i=0; i<num_strings; i++)
			strings[i].~SGString<T>();

		SG_FREE(strings);

		strings = NULL;
		num_strings = 0;
		max_string_length = 0;
	}

public:
	/** number of strings */
	index_t num_strings;

	/** length of longest string */
	index_t max_string_length;

	/** this contains the array of features */
	SGString<T>* strings;
};
}
#endif // __SGSTRINGLIST_H__
