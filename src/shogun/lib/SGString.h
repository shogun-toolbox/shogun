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

#include <shogun/lib/config.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGReferencedData.h>

namespace shogun
{

/** @brief shogun string */
template<class T> class SGString : public SGReferencedData
{
public:
	/** default constructor */
	SGString() : SGReferencedData()
	{
		init_data();
	}

	/** constructor for setting params */
	SGString(T* s, index_t l, bool ref_counting=true) : 
		SGReferencedData(ref_counting), 
		string(s), slen(l)
	{
	}

	/** constructor for setting params from a SGVector*/
	SGString(SGVector<T> v, bool ref_counting=true) :
		SGReferencedData(ref_counting),
		slen(v.vlen)
	{
		string = SG_MALLOC(T, slen);
		memcpy(string, v.vector, sizeof(T)*slen);
	}

	/** constructor to create new string in memory */
	SGString(index_t len, bool ref_counting=true) :
		slen(len)
	{
		string=SG_MALLOC(T, len);
	}

	/** copy constructor */
	SGString(const SGString &orig) : 
		SGReferencedData(orig)
	{
		copy_data(orig);
	}

	/** destructor */
	virtual ~SGString() 
	{
		unref();
	}

	/** equality operator */
	bool operator==(const SGString & other) const
	{
		if (other.slen != slen)
			return false;

		for (int i = 0; i < slen; i++)
		{
			if (other.string[i] != string[i])
				return false;
		}

		return true;
	}

protected:

	/** copy data */
	virtual void copy_data(const SGReferencedData &orig)
	{
		string = ((SGString*)(&orig))->string;
		slen = ((SGString*)(&orig))->slen;
	}

	/** init data */
	virtual void init_data()
	{
		string = NULL;
		slen = 0;
	}

	/** free string */
	virtual void free_data()
	{
		SG_FREE(string);
		string = NULL;
		slen = 0;
	}

public:

	/** string  */
	T* string;
	/** length of string  */
	index_t slen;
};
}
#endif // __SGSTRING_H__
