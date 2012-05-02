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

namespace shogun
{
/** @brief template class SGStringList */
template <class T> struct SGStringList
{
public:
	/** default constructor */
	SGStringList() : num_strings(0), max_string_length(0), strings(NULL),
		do_free(false) { }

	/** constructor for setting params */
	SGStringList(SGString<T>* s, index_t num_s, index_t max_length,
			bool free_strings=false) : num_strings(num_s),
			max_string_length(max_length), strings(s), do_free(free_strings) { }

	/** constructor to create new string list in memory */
	SGStringList(index_t num_s, index_t max_length, bool free_strings=false)
		: num_strings(num_s), max_string_length(max_length),
		  do_free(free_strings)
	{
		strings=SG_MALLOC(SGString<T>, num_strings);
	}

	/** copy constructor */
	SGStringList(const SGStringList &orig) :
		num_strings(orig.num_strings),
		max_string_length(orig.max_string_length),
		strings(orig.strings), do_free(orig.do_free) { }

	/** free list */
	void free_list()
	{
		if (do_free)
			SG_FREE(strings);

		strings=NULL;
		do_free=false;
		num_strings=0;
		max_string_length=0;
	}

	/** destroy list */
	void destroy_list()
	{
		do_free=true;
		free_list();
	}

public:
	/** number of strings */
	index_t num_strings;

	/** length of longest string */
	index_t max_string_length;

	/** this contains the array of features */
	SGString<T>* strings;

	/** whether vector needs to be freed */
	bool do_free;
};
}
#endif // __SGSTRINGLIST_H__
