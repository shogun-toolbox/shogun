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
#ifndef __SGSTRING_H__
#define __SGSTRING_H__

#include <shogun/lib/config.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

/** @brief shogun string */
template<class T> class SGString
{
public:
	/** default constructor */
	SGString() : string(NULL), slen(0), do_free(false) { }

	/** constructor for setting params */
	SGString(T* s, index_t l, bool free_s=false)
		: string(s), slen(l), do_free(free_s) { }

	/** constructor for setting params from a SGVector*/
	SGString(SGVector<T> v)
		: string(v.vector), slen(v.vlen), do_free(v.do_free) { }

	/** constructor to create new string in memory */
	SGString(index_t len, bool free_s=false) :
		slen(len), do_free(free_s)
	{
		string=SG_MALLOC(T, len);
	}

	/** copy constructor */
	SGString(const SGString &orig)
		: string(orig.string), slen(orig.slen), do_free(orig.do_free) { }

	/** free string */
	void free_string()
	{
		if (do_free)
			SG_FREE(string);

		string=NULL;
		do_free=false;
		slen=0;
	}

	/** destroy string */
	void destroy_string()
	{
		do_free=true;
		free_string();
	}

public:
	/** string  */
	T* string;
	/** length of string  */
	index_t slen;
	/** whether string needs to be freed */
	bool do_free;
};
}
#endif // __SGSTRING_H__
