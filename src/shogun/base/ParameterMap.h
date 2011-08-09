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
	 * @param ctype container type of parameter
	 * @param stype struct type of parameter
	 * @param ptype primitive type of parameter
	 */
	CSGParamInfo(const char* name, EContainerType ctype, EStructType stype,
			EPrimitiveType ptype);

	/** destructor */
	virtual ~CSGParamInfo();

	/** prints all parameter values */
	void print_param_info();

	/** operator for comparison, true iff all attributes are equal */
	bool operator==(const CSGParamInfo& other) const;

	/** operator for comparison (by string m_name) */
	bool operator<(const CSGParamInfo& other) const;

	/** operator for comparison (by string m_name) */
	bool operator>(const CSGParamInfo& other) const;

	/** @return name of the SG_SERIALIZABLE */
	inline virtual const char* get_name() const { return "SGParamInfo";	}

private:
	void init();

public:
	char* m_name;
	EContainerType m_ctype;
	EStructType m_stype;
	EPrimitiveType m_ptype;
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

	/** operator for comparison, true iff m_key is equal */
	bool operator==(const CParameterMapElement& other) const;

	/** operator for comparison (by m_key) */
	bool operator<(const CParameterMapElement& other) const;

	/** operator for comparison (by m_key) */
	bool operator>(const CParameterMapElement& other) const;

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
 * Implementation is done via an array. Via the call finalize_map(), it is
 * sorted. Then, get() may be called. If it is called before, an error is
 * thrown.
 *
 * In finalize_map() the array is sorted.
 * So inserting n elements is n*O(1) + O(n*log n) = O(n*log n).
 * Getting an element is then possible in O(log n) by binary search
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
	 * finalize_map() has to be called first if more than one elements are in
	 * map
	 *
	 * @param key key of the element to get
	 * @return value of the key element
	 */
	CSGParamInfo* get(CSGParamInfo* key) const;

	/** Finalizes the map. Has to be called before get may be called if more
	 * than one element in map */
	void finalize_map();

	/** prints all elements of this map */
	void print_map();

	/** @return name of the SG_SERIALIZABLE */
	inline virtual const char* get_name() const { return "ParameterMap"; }

private:
	void init();

protected:
	/** list of CLinearMap elements, this is always kept sorted */
	CDynamicObjectArray<CParameterMapElement>* m_map_elements;

	/** variable that indicates if underlying array is sorted (and thus get
	 * may safely be called) */
	bool m_finalized;
};

}

#endif /* __PARAMETERMAP_ */
