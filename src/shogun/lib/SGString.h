/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Written (W) 2010,2012 Soeren Sonnenburg
 * Written (W) 2012 Jacob Walker
 * Copyright (C) 2010 Berlin Institute of Technology
 * Copyright (C) 2012 Soeren Sonnenburg
 */
#ifndef __SGSTRING_H__
#define __SGSTRING_H__

#include <lib/config.h>
#include <lib/DataType.h>

namespace shogun
{

template<class T> class SGVector;
class CFile;

/** @brief shogun string */
template<class T> class SGString
{
public:
	/** default constructor */
	SGString();

	/** constructor for setting params */
	SGString(T* s, index_t l, bool free_s=false);

	/** constructor for setting params from a SGVector*/
	SGString(SGVector<T> v);

	/** constructor to create new string in memory */
	SGString(index_t len, bool free_s=false);

	/** copy constructor */
	SGString(const SGString &orig);

	/** equality operator */
	bool operator==(const SGString & other) const;

	/** free string */
	void free_string();

	/** destroy string */
	void destroy_string();

	/**
	 * get the string (no copying is done here)
	 *
	 * @return the refcount increased string
	 */
	inline SGString<T> get()
	{
		return *this;
	}

	/** load string from file
	 *
	 * @param loader File object via which to load data
	 */
	void load(CFile* loader);

	/** save string to file
	 *
	 * @param saver File object via which to save data
	 */
	void save(CFile* saver);

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
