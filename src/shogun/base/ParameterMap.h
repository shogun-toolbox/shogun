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

#include <shogun/lib/config.h>

#include <shogun/base/DynArray.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/common.h>

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
	void print_param_info(const char* prefix="") const;

	/** @return string representation, caller has to clean up */
	char* to_string() const;

	/** @return an identical copy */
	SGParamInfo* duplicate() const;

	/** operator for comparison, true iff all attributes are equal */
	bool operator==(const SGParamInfo& other) const;

	/** operator for comparison, false iff all attributes are equal */
	bool operator!=(const SGParamInfo& other) const;

	/** operator for comparison (by string m_name, if equal by others) */
	bool operator<(const SGParamInfo& other) const;

	/** operator for comparison (by string m_name, if equal by others) */
	bool operator>(const SGParamInfo& other) const;

	/** @return true iff this was constructed using the std constructor (empty
	 * parameter used to say that it appeared here first time */
	bool is_empty() const;

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
 * key and a set of values, which each are of type SGParamInfo.
 * May be compared to each other based on their keys
 */
class ParameterMapElement
{
public:
	/** constructor */
	ParameterMapElement();

	/** constructor
	 *
	 * @param key key of this element
	 * @param values array of value of this element
	 */
	ParameterMapElement(const SGParamInfo* key,
			DynArray<const SGParamInfo*>* values);

	/** destructor */
	virtual ~ParameterMapElement();

	/** operator for comparison, true iff m_key is equal */
	bool operator==(const ParameterMapElement& other) const;

	/** operator for comparison (by m_key) */
	bool operator<(const ParameterMapElement& other) const;

	/** operator for comparison (by m_key) */
	bool operator>(const ParameterMapElement& other) const;

	/** @return name of the SG_SERIALIZABLE */
	virtual const char* get_name() const
	{
		return "ParameterMapElement";
	}

public:
	/** key */
	const SGParamInfo* m_key;

	/** values */
	DynArray<const SGParamInfo*>* m_values;

};

/** @brief Implements a map of ParameterMapElement instances
 * Maps one key to a set of values.
 *
 * Implementation is done via an array. Via the call finalize_map(), a hidden
 * structure is built which bundles all values for each key.
 * Then, get() may be called, which returns an array of values for a key.
 * If it is called before, an error is
 * thrown.
 *
 * Putting elements is in O(1).
 * finalize_map sorts the underlying array and then regroups values, O(n*log n).
 * Add all values and then call once.
 * Getting a set of values is possible in O(log n) via binary search
 */
class ParameterMap
{
public:
	/** constructor */
	ParameterMap();

	/** destructor */
	virtual ~ParameterMap();

	/** Puts an newly allocated element into the map. Note that get(...)
	 * returns an array of value elements, so it is allowed to add multiple
	 * values for one key. Note that there is also no check for double entries,
	 * i.e. same key and same value.This will result in two elements when get
	 * is called.
	 * Operation in O(1).
	 *
	 * @param key key of the element
	 * @param value value of the lement
	 */
	void put(const SGParamInfo* key, const SGParamInfo* value);

	/** Gets a specific element of the map
	 * finalize_map() has to be called first if more than one elements are in
	 * map.
	 *
	 * Operation in O(log n)
	 *
	 * Same as below but without pointer for syntactic ease.
	 *
	 * parameter key: key of the element to get
	 * returns set of values of the key element
	 */
	DynArray<const SGParamInfo*>* get(const SGParamInfo) const;

	/** Gets a specific element of the map.
	 * finalize_map() has to be called first if more than one elements are in
	 * map.
	 *
	 * Operation in O(log n)
	 *
	 * @param key key of the element to get
	 * @return set of values of the key element
	 */
	DynArray<const SGParamInfo*>* get(const SGParamInfo* key) const;

	/** Finalizes the map. Has to be called before get may be called if more
	 * than one element in map
	 *
	 * Operation in O(n*log n)
	 */
	void finalize_map();

	/** prints all elements of this map */
	void print_map();

protected:
	/** list of CLinearMap elements, this is always kept sorted */
	DynArray<ParameterMapElement*> m_map_elements;

	/** hidden internal structure which is used to hold multiple values for one
	 * key. It is built when finalize_map() is called. */
	DynArray<ParameterMapElement*> m_multi_map_elements;

	/** variable that indicates if its possible to call get method */
	bool m_finalized;
};

}

#endif /* __PARAMETERMAP_ */
