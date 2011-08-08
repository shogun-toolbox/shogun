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

CSGParamInfo::CSGParamInfo(const char* name)
{
	init();

	/* copy name */
	m_name=SG_MALLOC(char, strlen(name)+1);
	strcpy(m_name, name);
}

CSGParamInfo::~CSGParamInfo()
{
	SG_FREE(m_name);
}

void CSGParamInfo::print()
{
	SG_PRINT("%s with: ", get_name());
	SG_SPRINT("name=\"%s\"", m_name);
	SG_PRINT("\n");
}

void CSGParamInfo::init()
{
	m_name=NULL;
	m_parameters->add(m_name, "name", "Name of parameter");
//	m_parameters->add(m_type, "type", "Type of parameter");
}

bool CSGParamInfo::operator==(const CSGParamInfo& other) const
{
	return !strcmp(m_name, other.m_name);
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
