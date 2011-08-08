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

CSGParamInfo::CSGParamInfo()
{
	init();
}

CSGParamInfo::CSGParamInfo(const char* name, EContainerType ctype,
		EStructType stype, EPrimitiveType ptype)
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

void CSGParamInfo::print()
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

CParameterMapElement::CParameterMapElement()
{
	init();
}

CParameterMapElement::CParameterMapElement(CSGParamInfo* key,
		CSGParamInfo* value)
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

void CParameterMapElement::init()
{
	m_key=NULL;
	m_value=NULL;

	m_parameters->add((CSGObject**)&m_key, "key", "Key of map element");
	m_parameters->add((CSGObject**)&m_value, "value", "Value of map element");
}

CParameterMap::CParameterMap()
{
	m_map_elements=new CDynamicObjectArray<CParameterMapElement>();
	SG_REF(m_map_elements);
}

CParameterMap::~CParameterMap()
{
	SG_UNREF(m_map_elements);
}

void CParameterMap::put(CSGParamInfo* key, CSGParamInfo* value)
{
	m_map_elements->append_element(new CParameterMapElement(key, value));
}

CSGParamInfo* CParameterMap::get(CSGParamInfo* key) const
{
	/* perform linear search to find corresponding value */
	index_t i;
	for (i=0; i<m_map_elements->get_num_elements(); ++i)
	{
		CParameterMapElement* current=m_map_elements->get_element(i);
		if (*current->m_key==*key)
		{
			CSGParamInfo* value=current->m_value;
			SG_UNREF(current);
			SG_REF(value);
			return value;
		}

		SG_UNREF(current);
	}

	return NULL;
}
