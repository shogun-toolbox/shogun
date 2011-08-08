/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __PARAMETERMAP_
#define __PARAMETERMAP_

#include <shogun/lib/DynamicObjectArray.h>

namespace shogun
{

/** @brief Class that holds informations about a certain parameter of an
 * CSGObject. Contains name, type, etc.
 * This is used for mapping types that have changed in different versions of
 * shogun.
 * Instances of this class may be compared to each other using == and the
 * parameter's name is used for comparison.
 */
class CSGParamInfo: public CSGObject
{
public:
	/** constructor */
	CSGParamInfo();

	/** constructor
	 *
	 * @param name name of parameter, is copied
	 */
	CSGParamInfo(const char* name);

	/** destructor */
	virtual ~CSGParamInfo();

	/** prints all parameter values */
	void print();

	/** operator for comparison, true iff m_name is equal */
	bool operator==(const CSGParamInfo& other) const;

	/** @return name of the SG_SERIALIZABLE */
	inline virtual const char* get_name() const { return "SGParamInfo";	}

private:
	void init();

public:
	char* m_name;

};

/** @brief Class to hold instances of a parameter map. Each element contains a
 * key and a value, which are of type CSGParamInfo.
 */
class CParameterMapElement: public CSGObject
{
public:
	/** constructor */
	CParameterMapElement();

	/** constructor
	 *
	 * @param key key of this element
	 * @param value value of this element
	 */
	CParameterMapElement(CSGParamInfo* key, CSGParamInfo* value);

	/** destructor */
	virtual ~CParameterMapElement();

	/** @return name of the SG_SERIALIZABLE */
	inline virtual const char* get_name() const
	{
		return "ParameterMapElement";
	}

private:
	void init();

public:
	CSGParamInfo* m_key;
	CSGParamInfo* m_value;

};

/** @brief Implements a map of CParameterMapElement instances
 *
 * Implementation is simple: O(n) for get, O(1) for put.
 * Beware of putting large amounts of elements in this map! Slow.
 */
class CParameterMap: public CSGObject
{
public:
	/** constructor */
	CParameterMap();

	/** destructor */
	virtual ~CParameterMap();

	/** Puts an newly allocated element into the map
	 *
	 * @param key key of the element
	 * @param value value of the lement
	 */
	void put(CSGParamInfo* key, CSGParamInfo* value);

	/** Gets a specific element of the map. Note that it is SG_REF'ed
	 *
	 * @param key key of the element to get
	 * @return value of the key element
	 */
	CSGParamInfo* get(CSGParamInfo* key) const;

	/** @return name of the SG_SERIALIZABLE */
	inline virtual const char* get_name() const { return "ParameterMap"; }

protected:
	/** list of CLinearMap elements, this is always kept sorted */
	CDynamicObjectArray<CParameterMapElement>* m_map_elements;
};

}

#endif /* __PARAMETERMAP_ */
