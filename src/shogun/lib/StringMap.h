/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __CSTRINGMAP_H_
#define __CSTRINGMAP_H_

#include <shogun/base/SGObject.h>

namespace shogun
{

class CStringMapElement : public CSGObject
{
public:
	CStringMapElement();

	CStringMapElement(const char* key, const char* value);

	virtual ~CStringMapElement();

	inline virtual const char* get_name() const { return "StringMapElement"; }

private:
	void init();

public:
	char* m_key;
	char* m_value;
};

template<class T> class CDynamicObjectArray;

class CStringMap : public CSGObject
{
public:
	CStringMap();

	virtual ~CStringMap();

	void put(const char* key, const char* value);
	const char* get(const char* key) const;

	inline virtual const char* get_name() const { return "StringMap"; }

protected:
	/** list of CStringMap elements, this is always kept sorted */
	CDynamicObjectArray<CStringMapElement>* m_map_elements;
};

}

#endif /* __CSTRINGMAP_H_ */
