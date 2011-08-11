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

SGParamInfo::SGParamInfo()
{
	init();
}

SGParamInfo::SGParamInfo(const char* name, EContainerType ctype,
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

SGParamInfo::~SGParamInfo()
{
	SG_FREE(m_name);
}

void SGParamInfo::print_param_info()
{
	SG_SPRINT("SGParamInfo with: ");

	SG_SPRINT("name=\"%s\"", m_name);

	TSGDataType t(m_ctype, m_stype, m_ptype);
	index_t buffer_length=100;
	char* buffer=SG_MALLOC(char, buffer_length);
	t.to_string(buffer, buffer_length);
	SG_SPRINT(", type=%s", buffer);
	SG_FREE(buffer);

	SG_SPRINT("\n");
}

SGParamInfo* SGParamInfo::duplicate() const
{
	return new SGParamInfo(m_name, m_ctype, m_stype, m_ptype);
}

void SGParamInfo::init()
{
	m_name=NULL;
	m_ctype=(EContainerType) 0;
	m_stype=(EStructType) 0;
	m_ptype=(EPrimitiveType) 0;
}

bool SGParamInfo::operator==(const SGParamInfo& other) const
{
	bool result=true;
	result&=!strcmp(m_name, other.m_name);
	result&=m_ctype==other.m_ctype;
	result&=m_stype==other.m_stype;
	result&=m_ptype==other.m_ptype;
	return result;
}

bool SGParamInfo::operator<(const SGParamInfo& other) const
{
	return strcmp(m_name, other.m_name)<0;
}

bool SGParamInfo::operator>(const SGParamInfo& other) const
{
	return strcmp(m_name, other.m_name)>0;
}

ParameterMapElement::ParameterMapElement()
{
	init();
}

ParameterMapElement::ParameterMapElement(SGParamInfo* key,
		SGParamInfo* value)
{
	init();

	m_key=key;
	m_value=value;
}

ParameterMapElement::~ParameterMapElement()
{
	delete m_key;
	delete m_value;
}

bool ParameterMapElement::operator==(const ParameterMapElement& other) const
{
	return *m_key==*other.m_key;
}

bool ParameterMapElement::operator<(const ParameterMapElement& other) const
{
	return *m_key<*other.m_key;
}

bool ParameterMapElement::operator>(const ParameterMapElement& other) const
{
	return *m_key>*other.m_key;
}

void ParameterMapElement::init()
{
	m_key=NULL;
	m_value=NULL;
}

ParameterMap::ParameterMap()
{
	init();
}

void ParameterMap::init()
{
	m_finalized=false;
}

ParameterMap::~ParameterMap()
{
	for (index_t i=0; i<m_map_elements.get_num_elements(); ++i)
		delete m_map_elements[i];
}

void ParameterMap::put(SGParamInfo* key, SGParamInfo* value)
{
	m_map_elements.append_element(new ParameterMapElement(key, value));
	m_finalized=false;
}

SGParamInfo* ParameterMap::get(SGParamInfo* key) const
{
	index_t num_elements=m_map_elements.get_num_elements();

	/* check if underlying array is sorted */
	if (!m_finalized && num_elements>1)
		SG_SERROR("Call finalize_map() before calling get()\n");

	/* do binary search in array of pointers */
	SGVector<ParameterMapElement*> array(m_map_elements.get_array(),
			num_elements);

	/* dummy element for searching */
	ParameterMapElement* dummy=new ParameterMapElement(key->duplicate(),
			key->duplicate());
	index_t index=CMath::binary_search<ParameterMapElement> (array, dummy);
	delete dummy;

	if (index==-1)
		return NULL;

	ParameterMapElement* element=m_map_elements.get_element(index);
	SGParamInfo* value=element->m_value;

	return value;
}

void ParameterMap::finalize_map()
{
	/* sort underlying array */
	SGVector<ParameterMapElement*> array(m_map_elements.get_array(),
			m_map_elements.get_num_elements());

	CMath::qsort<ParameterMapElement> (array);

	m_finalized=true;
}

void ParameterMap::print_map()
{
	for (index_t i=0; i< m_map_elements.get_num_elements(); ++i)
	{
		ParameterMapElement* current=m_map_elements[i];
		SG_SPRINT("%d\n", i);
		SG_SPRINT("key: ");
		current->m_key->print_param_info();
		SG_SPRINT("value: ");
		current->m_value->print_param_info();
	}
}
