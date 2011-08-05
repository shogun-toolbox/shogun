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

/** @brief Implements key-value pairs that are put into a CStringMap
 * Note that strings are copied in constructor.
 */
class CStringMapElement : public CSGObject
{
public:
	/** constructor */
	CStringMapElement();

	/** constructor. Takes key and value string and stores a copy of it */
	CStringMapElement(const char* key, const char* value);

	/** destructor */
	virtual ~CStringMapElement();

	/** @return name of the SG_SERIALIZABLE */
	inline virtual const char* get_name() const { return "StringMapElement"; }

private:
	void init();

public:
	char* m_key;
	char* m_value;
};

template<class T> class CDynamicObjectArray;

/** @brief implements a simple map for c-strings.
 * Note that these are copied when put to the map. When getting an element,
 * these copies are returned.
 *
 * As for now the implemenation is simple: O(n) for get, O(1) for put.
 * This may be done by some kind of tree to get logarithmic costs.
 * Beware of putting large amounts of elements in this map! Slow.
 */
class CStringMap : public CSGObject
{
public:
	/** constructor */
	CStringMap();

	/** destructor */
	virtual ~CStringMap();

	/** puts a key-value pair into the map
	 *
	 * @param key key of the element
	 * @param value value of the element
	 */
	void put(const char* key, const char* value);

	/** gets a value of a provided key
	 *
	 * @param key key of the element to get
	 * @return value of the specified key, NULL if there is no such
	 */
	const char* get(const char* key) const;

	/** @return name of the SG_SERIALIZABLE */
	inline virtual const char* get_name() const { return "StringMap"; }

protected:
	/** list of CStringMap elements, this is always kept sorted */
	CDynamicObjectArray<CStringMapElement>* m_map_elements;
};

}

#endif /* __CSTRINGMAP_H_ */
