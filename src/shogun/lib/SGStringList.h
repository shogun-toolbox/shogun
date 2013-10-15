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
#include <shogun/lib/common.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGReferencedData.h>
#include <shogun/lib/SGString.h>

namespace shogun
{

/** @brief template class SGStringList */
template <class T> class SGStringList : public SGReferencedData
{
public:
	/** default constructor */
	SGStringList();

	/** constructor for setting params */
	SGStringList(SGString<T>* s, index_t num_s, index_t max_length,
			bool ref_counting=true);

	/** constructor to create new string list in memory */
	SGStringList(index_t num_s, index_t max_length, bool ref_counting=true);

	/** copy constructor */
	SGStringList(const SGStringList &orig);

	/** destructor */
	virtual ~SGStringList();

	/**
	 * get the string list (no copying is done here)
	 *
	 * @return the refcount increased string list
	 */
	inline SGStringList<T> get()
	{
		return *this;
	}

	/** load strings from file
	 *
	 * @param loader File object via which to load data
	 */
	void load(CFile* loader);

	/** save strings to file
	 *
	 * @param saver File object via which to save data
	 */
	void save(CFile* saver);


protected:

	/** copy data */
	virtual void copy_data(const SGReferencedData &orig);

	/** init data */
	virtual void init_data();

	/** free data */
	void free_data();

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
