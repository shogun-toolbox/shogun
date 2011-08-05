/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */


#include <shogun/lib/StringMap.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

CStringMapElement::CStringMapElement()
{
	init();
}

CStringMapElement::CStringMapElement(const char* key, const char* value)
{
	init();

	/* copy strings (with respect to \0)*/
	m_key=SG_MALLOC(char, strlen(key)+1);
	m_value=SG_MALLOC(char, strlen(value)+1);
	strcpy(m_key, key);
	strcpy(m_value, value);
}

void CStringMapElement::init()
{
	m_key=NULL;
	m_value=NULL;

	m_parameters->add(m_key, "key", "Key of map element");
	m_parameters->add(m_value, "value", "Value of map element");
}

CStringMapElement::~CStringMapElement()
{
	SG_FREE(m_key);
	SG_FREE(m_value);
}

CStringMap::CStringMap()
{
	m_map_elements=new CDynamicObjectArray<CStringMapElement>();
	SG_REF(m_map_elements);

	m_parameters->add((CSGObject**)&m_map_elements,	"map_elements",
			"Array of map elements");
}

CStringMap::~CStringMap()
{
	SG_UNREF(m_map_elements);
}

void CStringMap::put(const char* key, const char* value)
{
	m_map_elements->append_element(new CStringMapElement(key, value));
}

const char* CStringMap::get(const char* key) const
{
	/* perform linear search to find corresponding value */
	index_t i;
	for (i=0; i<m_map_elements->get_num_elements(); ++i)
	{
		CStringMapElement* current=m_map_elements->get_element(i);
		if (!strcmp(key, current->m_key))
		{
			const char* value=current->m_value;
			SG_UNREF(current);
			return value;
		}

		SG_UNREF(current);
	}

	return NULL;
}
