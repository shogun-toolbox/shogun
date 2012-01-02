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

#include <shogun/base/DynArray.h>

namespace shogun
{

struct TParameter;

/** @brief Class that holds informations about a certain parameter of an
 * CSGObject. Contains name, type, etc.
 * This is used for mapping types that have changed in different versions of
 * shogun.
 * Instances of this class may be compared to each other. Ordering is based on
 * name, equalness is based on all attributes
 */
class SGParamInfo
{
public:
	/** constructor */
	SGParamInfo();

	/** constructor
	 *
	 * @param name name of parameter, is copied
	 * @param ctype container type of parameter
	 * @param stype struct type of parameter
	 * @param ptype primitive type of parameter
	 * @param param_version version of parameter
	 */
	SGParamInfo(const char* name, EContainerType ctype, EStructType stype,
			EPrimitiveType ptype, int32_t param_version);

	/** constructor to create from a TParameter instance
	 *
	 * @param param TParameter instance to use
	 * @param param_version version of parameter
	 */
	SGParamInfo(const TParameter* param, int32_t param_version);

	/** copy constructor
	 * @param orig element to copy from
	 */
	SGParamInfo(const SGParamInfo& orig);

	/** destructor */
	virtual ~SGParamInfo();

	/** prints all parameter values */
	void print_param_info();

	/** @return string representation, caller has to clean up */
	char* to_string() const;

	/** @return an identical copy */
	SGParamInfo* duplicate() const;

	/** operator for comparison, true iff all attributes are equal */
	bool operator==(const SGParamInfo& other) const;

	/** operator for comparison (by string m_name, if equal by others) */
	bool operator<(const SGParamInfo& other) const;

	/** operator for comparison (by string m_name, if equal by others) */
	bool operator>(const SGParamInfo& other) const;

private:
	void init();

public:
	/** name */
	char* m_name;

	/** container type */
	EContainerType m_ctype;

	/** struct type */
	EStructType m_stype;

	/** primitive type */
	EPrimitiveType m_ptype;

	/** version of the parameter */
	int32_t m_param_version;
};

/** @brief Class to hold instances of a parameter map. Each element contains a
 * key and a value, which are of type SGParamInfo.
 * May be compared to each other based on their keys
 */
class ParameterMapElement
{
public:
	/** constructor */
	ParameterMapElement();

	/** constructor
	 *
	 * @param key key of this element, is copied
	 * @param value value of this element, is copied
	 */
	ParameterMapElement(SGParamInfo* key, SGParamInfo* value);

	/** destructor */
	virtual ~ParameterMapElement();

	/** operator for comparison, true iff m_key is equal */
	bool operator==(const ParameterMapElement& other) const;

	/** operator for comparison (by m_key) */
	bool operator<(const ParameterMapElement& other) const;

	/** operator for comparison (by m_key) */
	bool operator>(const ParameterMapElement& other) const;

	/** @return name of the SG_SERIALIZABLE */
	inline virtual const char* get_name() const
	{
		return "ParameterMapElement";
	}

private:
	void init();

public:
	/** keys */
	SGParamInfo* m_key;
	/** values */
	SGParamInfo* m_value;

};

/** @brief Implements a map of ParameterMapElement instances
 *
 * Implementation is done via an array. Via the call finalize_map(), it is
 * sorted. Then, get() may be called. If it is called before, an error is
 * thrown.
 *
 * In finalize_map() the array is sorted.
 * So inserting n elements is n*O(1) + O(n*log n) = O(n*log n).
 * Getting an element is then possible in O(log n) by binary search
 */
class ParameterMap
{
public:
	/** constructor */
	ParameterMap();

	/** destructor */
	virtual ~ParameterMap();

	/** Puts an newly allocated element into the map
	 *
	 * @param key key of the element
	 * @param value value of the lement
	 */
	void put(SGParamInfo* key, SGParamInfo* value);

	/** Gets a specific element of the map. Note that it is SG_REF'ed
	 * finalize_map() has to be called first if more than one elements are in
	 * map
	 *
	 * @param key key of the element to get
	 * @return value of the key element
	 */
	SGParamInfo* get(SGParamInfo* key) const;

	/** Finalizes the map. Has to be called before get may be called if more
	 * than one element in map */
	void finalize_map();

	/** prints all elements of this map */
	void print_map();

private:
	void init();

protected:
	/** list of CLinearMap elements, this is always kept sorted */
	DynArray<ParameterMapElement*> m_map_elements;

	/** variable that indicates if underlying array is sorted (and thus get
	 * may safely be called) */
	bool m_finalized;
};

}

#endif /* __PARAMETERMAP_ */
