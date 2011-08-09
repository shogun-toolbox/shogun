/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/ParameterMap.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CSGParamInfo::CSGParamInfo() : CSGObject()
{
	init();
}

CSGParamInfo::CSGParamInfo(const char* name, EContainerType ctype,
		EStructType stype, EPrimitiveType ptype) : CSGObject()
{
	init();

	/* copy name */
	m_name=SG_MALLOC(char, strlen(name)+1);
	strcpy(m_name, name);

	m_ctype=ctype;
	m_stype=stype;
	m_ptype=ptype;
}

CSGParamInfo::~CSGParamInfo()
{
	SG_FREE(m_name);
}

void CSGParamInfo::print_param_info()
{
	SG_PRINT("%s with: ", get_name());

	SG_SPRINT("name=\"%s\"", m_name);

	TSGDataType t(m_ctype, m_stype, m_ptype);
	index_t buffer_length=100;
	char* buffer=SG_MALLOC(char, buffer_length);
	t.to_string(buffer, buffer_length);
	SG_PRINT(", type=%s", buffer);
	SG_FREE(buffer);

	SG_PRINT("\n");
}

void CSGParamInfo::init()
{
	m_name=NULL;
	m_ctype=(EContainerType) 0;
	m_stype=(EStructType) 0;
	m_ptype=(EPrimitiveType) 0;

	m_parameters->add(m_name, "name", "Name of parameter");
	m_parameters->add((int*) &m_ctype, "ctype", "Container type of parameter");
	m_parameters->add((int*) &m_stype, "stype", "Structure type of parameter");
	m_parameters->add((int*) &m_ptype, "ptype", "Primitive type of parameter");
}

bool CSGParamInfo::operator==(const CSGParamInfo& other) const
{
	bool result=true;
	result&=!strcmp(m_name, other.m_name);
	result&=m_ctype==other.m_ctype;
	result&=m_stype==other.m_stype;
	result&=m_ptype==other.m_ptype;
	return result;
}

bool CSGParamInfo::operator<(const CSGParamInfo& other) const
{
	return strcmp(m_name, other.m_name)<0;
}

bool CSGParamInfo::operator>(const CSGParamInfo& other) const
{
	return strcmp(m_name, other.m_name)>0;
}

CParameterMapElement::CParameterMapElement() : CSGObject()
{
	init();
}

CParameterMapElement::CParameterMapElement(CSGParamInfo* key,
		CSGParamInfo* value) : CSGObject()
{
	init();

	m_key=key;
	m_value=value;

	SG_REF(m_key);
	SG_REF(m_value);
}

CParameterMapElement::~CParameterMapElement()
{
	SG_UNREF(m_key);
	SG_UNREF(m_value);
}

bool CParameterMapElement::operator==(const CParameterMapElement& other) const
{
	return *m_key==*other.m_key;
}

bool CParameterMapElement::operator<(const CParameterMapElement& other) const
{
	return *m_key<*other.m_key;
}

bool CParameterMapElement::operator>(const CParameterMapElement& other) const
{
	return *m_key>*other.m_key;
}

void CParameterMapElement::init()
{
	m_key=NULL;
	m_value=NULL;

	m_parameters->add((CSGObject**)&m_key, "key", "Key of map element");
	m_parameters->add((CSGObject**)&m_value, "value", "Value of map element");
}

CParameterMap::CParameterMap() : CSGObject()
{
	init();
}

void CParameterMap::init()
{
	m_map_elements=new CDynamicObjectArray<CParameterMapElement>();
	SG_REF(m_map_elements);

	m_finalized=false;

	m_parameters->add((CSGObject**)&m_map_elements, "map_elements",
			"Array of map elements");
	m_parameters->add(&m_finalized, "finalized", "Whether map is finalized");
}

CParameterMap::~CParameterMap()
{
	SG_UNREF(m_map_elements);
}

void CParameterMap::put(CSGParamInfo* key, CSGParamInfo* value)
{
	m_map_elements->append_element(new CParameterMapElement(key, value));
	m_finalized=false;
}

CSGParamInfo* CParameterMap::get(CSGParamInfo* key) const
{
	index_t num_elements=m_map_elements->get_num_elements();

	/* check if underlying array is sorted */
	if (!m_finalized && num_elements>1)
		SG_ERROR("Call finalize_map() before calling get()\n");

	/* do binary search in array of pointers */
	SGVector<CParameterMapElement*> array(m_map_elements->get_array(),
			num_elements);

	/* dummy element for searching */
	CParameterMapElement* dummy=new CParameterMapElement(key, key);
	index_t index=CMath::binary_search<CParameterMapElement> (array, dummy);
	SG_UNREF(dummy);

	if (index==-1)
		return NULL;

	CParameterMapElement* element=m_map_elements->get_element(index);
	CSGParamInfo* value=element->m_value;
	SG_REF(value);
	SG_UNREF(element);

	return value;
}

void CParameterMap::finalize_map()
{
	/* sort underlying array */
	SGVector<CParameterMapElement*> array(m_map_elements->get_array(),
			m_map_elements->get_num_elements());

	CMath::qsort<CParameterMapElement> (array);

	m_finalized=true;
}

void CParameterMap::print_map()
{
	for (index_t i=0; i< m_map_elements->get_num_elements(); ++i)
	{
		CParameterMapElement* current=m_map_elements->get_element(i);
		SG_PRINT("%d\n", i);
		SG_PRINT("key: ");
		current->m_key->print_param_info();
		SG_PRINT("value: ");
		current->m_value->print_param_info();
		SG_UNREF(current);
	}
}
