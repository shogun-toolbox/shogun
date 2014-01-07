/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <base/ParameterMap.h>
#include <base/Parameter.h>
#include <lib/memory.h>
#include <mathematics/Math.h>
#include <lib/DataType.h>

using namespace shogun;

SGParamInfo::SGParamInfo()
{
	m_name=NULL;
	m_ctype=CT_UNDEFINED;
	m_stype=ST_UNDEFINED;
	m_ptype=PT_UNDEFINED;
	m_param_version=-1;
}

SGParamInfo::SGParamInfo(const SGParamInfo& orig)
{
	/* copy name if existent */
	m_name=get_strdup(orig.m_name);

	m_ctype=orig.m_ctype;
	m_stype=orig.m_stype;
	m_ptype=orig.m_ptype;
	m_param_version=orig.m_param_version;
}

SGParamInfo::SGParamInfo(const char* name, EContainerType ctype,
		EStructType stype, EPrimitiveType ptype, int32_t param_version)
{
	/* copy name if existent */
	m_name=get_strdup(name);

	m_ctype=ctype;
	m_stype=stype;
	m_ptype=ptype;
	m_param_version=param_version;
}

SGParamInfo::SGParamInfo(const TParameter* param, int32_t param_version)
{
	/* copy name if existent */
	m_name=get_strdup(param->m_name);

	TSGDataType type=param->m_datatype;
	m_ctype=type.m_ctype;
	m_stype=type.m_stype;
	m_ptype=type.m_ptype;
	m_param_version=param_version;
}

SGParamInfo::~SGParamInfo()
{
	SG_FREE(m_name);
}

char* SGParamInfo::to_string() const
{
	char* buffer=SG_MALLOC(char, 200);
	strcpy(buffer, "SGParamInfo with: ");
	strcat(buffer, "name=\"");
	strcat(buffer, m_name ? m_name : "NULL");
	strcat(buffer, "\", type=");

	char* b;
	/* only cat type if it is defined (is not when std constructor was used)*/
	if (!is_empty())
	{
		TSGDataType t(m_ctype, m_stype, m_ptype);
		index_t l=100;
		b=SG_MALLOC(char, l);
		t.to_string(b, l);
		strcat(buffer, b);
		SG_FREE(b);
	}
	else
		strcat(buffer, "no type");

	b=SG_MALLOC(char, 10);
	sprintf(b, "%d", m_param_version);
	strcat(buffer, ", version=");
	strcat(buffer, b);
	SG_FREE(b);

	return buffer;
}

void SGParamInfo::print_param_info(const char* prefix) const
{
	char* s=to_string();
	SG_SPRINT("%s%s\n", prefix, s)
	SG_FREE(s);
}

SGParamInfo* SGParamInfo::duplicate() const
{
	return new SGParamInfo(m_name, m_ctype, m_stype, m_ptype, m_param_version);
}

bool SGParamInfo::operator==(const SGParamInfo& other) const
{
	bool result=true;

	/* handle NULL strings */
	if ((!m_name && other.m_name) || (m_name && !other.m_name))
		return false;

	if (m_name && other.m_name)
		result&=!strcmp(m_name, other.m_name);

	result&=m_ctype==other.m_ctype;
	result&=m_stype==other.m_stype;
	result&=m_ptype==other.m_ptype;
	result&=m_param_version==other.m_param_version;
	return result;
}

bool SGParamInfo::operator!=(const SGParamInfo& other) const
{
	return !operator ==(other);
}

bool SGParamInfo::operator<(const SGParamInfo& other) const
{
	/* NULL here is always smaller than anything */
	if (!m_name)
	{
		if (!other.m_name)
			return false;
		else
			return true;
	}
	else if (!other.m_name)
		return true;

	int32_t result=strcmp(m_name, other.m_name);

	if (result==0)
	{
		if (m_param_version==other.m_param_version)
		{
			if (m_ctype==other.m_ctype)
			{
				if (m_stype==other.m_stype)
				{
					if (m_ptype==other.m_ptype)
					{
						return false;
					}
					else
						return m_ptype<other.m_ptype;
				}
				else
					return m_stype<other.m_stype;
			}
			else
				return m_ctype<other.m_ctype;
		}
		else
			return m_param_version<other.m_param_version;

	}
	else
		return result<0;
}

bool SGParamInfo::operator>(const SGParamInfo& other) const
{
	return !(*this<(other)) && !(*this==other);
}

bool SGParamInfo::is_empty() const
{
	/* return true if this info is for empty parameter */
	return m_ctype==CT_UNDEFINED && m_stype==ST_UNDEFINED && m_ptype==PT_UNDEFINED && !m_name;
}

ParameterMapElement::ParameterMapElement()
{
	m_key=NULL;
	m_values=NULL;
}

ParameterMapElement::ParameterMapElement(const SGParamInfo* key,
		DynArray<const SGParamInfo*>* values)
{
	m_key=key;
	m_values=values;
}

ParameterMapElement::~ParameterMapElement()
{
	delete m_key;

	if (m_values)
	{
		for (index_t i=0; i<m_values->get_num_elements(); ++i)
			delete m_values->get_element(i);

		delete m_values;
	}
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

/*
  Initializing m_map_elements(1), m_multi_map_elements(1) with small
  preallocation-size, because ParameterMap will be constructed several
  times for EACH SGObject instance.
*/
ParameterMap::ParameterMap()
: m_map_elements(1), m_multi_map_elements(1)
{
	m_finalized=false;
}

ParameterMap::~ParameterMap()
{
	for (index_t i=0; i<m_map_elements.get_num_elements(); ++i)
		delete m_map_elements[i];

	for (index_t i=0; i<m_multi_map_elements.get_num_elements(); ++i)
		delete m_multi_map_elements[i];
}

void ParameterMap::put(const SGParamInfo* key, const SGParamInfo* value)
{
	/* assert that versions do differ exactly one if mapping is non-empty */
	if (key->m_param_version-value->m_param_version!=1)
	{
		if (!key->is_empty() && !value->is_empty())
		{
			char* s=key->to_string();
			char* t=value->to_string();
			SG_SERROR("Versions of parameter mappings from \"%s\" to \"%s\" have"
					" to differ exactly one\n", s, t);
			SG_FREE(s);
			SG_FREE(t);
		}
	}

	/* always add array of ONE element as values, will be processed later
	 * in finalize map method */
	DynArray<const SGParamInfo*>* values=new DynArray<const SGParamInfo*>();
	values->append_element(value);
	m_map_elements.append_element(new ParameterMapElement(key, values));
	m_finalized=false;
}

DynArray<const SGParamInfo*>* ParameterMap::get(const SGParamInfo key) const
{
	return get(&key);
}

DynArray<const SGParamInfo*>* ParameterMap::get(const SGParamInfo* key) const
{
	index_t num_elements=m_multi_map_elements.get_num_elements();

	/* check if maps is finalized */
	if (!m_finalized && num_elements)
		SG_SERROR("Call finalize_map() before calling get()\n")

	/* do binary search in array of pointers */
	/* dummy element for searching */
	ParameterMapElement* dummy=new ParameterMapElement(key->duplicate(), NULL);
	index_t index=CMath::binary_search<ParameterMapElement> (
			m_multi_map_elements.get_array(), num_elements, dummy);
	delete dummy;

	if (index==-1)
		return NULL;

	ParameterMapElement* element=m_multi_map_elements.get_element(index);
	return element->m_values;
}

void ParameterMap::finalize_map()
{
	/* only do something if there are elements in map */
	if (!m_map_elements.get_num_elements())
		return;

	/* sort underlying array */
	CMath::qsort<ParameterMapElement> (m_map_elements.get_array(),
			m_map_elements.get_num_elements());

//	SG_SPRINT("map elements before finalize\n")
//	for (index_t i=0; i<m_map_elements.get_num_elements(); ++i)
//	{
//		ParameterMapElement* current=m_map_elements[i];
//		SG_SPRINT("element %d:\n", i)
//		SG_SPRINT("\tkey: ")
//		current->m_key->print_param_info();
//		SG_SPRINT("\t%d values:\n", current->m_values->get_num_elements())
//		for (index_t j=0; j<current->m_values->get_num_elements(); ++j)
//			current->m_values->get_element(j)->print_param_info("\t\t");
//	}

	/* clear old multi elements. These were copies. */
	for (index_t i=0; i<m_multi_map_elements.get_num_elements(); ++i)
		delete m_multi_map_elements[i];

	m_multi_map_elements.reset(NULL);
//	SG_SPRINT("\nstarting finalization\n")

	/* iterate over all elements of map elements (have all one value (put)) and
	 * add all values of same key to ONE map element of hidden structure */
	DynArray<const SGParamInfo*>* values=new DynArray<const SGParamInfo*>();
	const SGParamInfo* current_key=m_map_elements[0]->m_key;
//	char* s=current_key->to_string();
//	SG_SPRINT("current key: %s\n", s)
//	SG_FREE(s);
	for (index_t i=0; i<m_map_elements.get_num_elements(); ++i)
	{
		const ParameterMapElement* current=m_map_elements[i];
		if (*current_key != *current->m_key)
		{
			/* create new values array to add and update key */
			values=new DynArray<const SGParamInfo*>();
			current_key=current->m_key;
//			s=current_key->to_string();
//			SG_SPRINT("new current key: %s\n", s)
//			SG_FREE(s);
		}

		/* add to values array */
		char* t=current->m_values->get_element(0)->to_string();
//		SG_SPRINT("\tadding %s\n", t)
		SG_FREE(t);
		values->append_element(current->m_values->get_element(0)->duplicate());

		/* if current values array has not been added to multi map elements, do
		 * now */
		index_t last_idx=m_multi_map_elements.get_num_elements()-1;
		if (last_idx<0 ||
				m_multi_map_elements.get_element(last_idx)->m_values != values)
		{
//			SG_SPRINT("adding values array\n")
			m_multi_map_elements.append_element(
				new ParameterMapElement(current_key->duplicate(), values));
		}
	}

	m_finalized=true;
//	SG_SPRINT("leaving finalize_map()\n")
}

void ParameterMap::print_map()
{
	/* check if maps is finalized */
	if (!m_finalized && m_map_elements.get_num_elements())
		SG_SERROR("Call finalize_map() before calling print_map()\n")

//	SG_SPRINT("map with %d keys:\n", m_multi_map_elements.get_num_elements())
	for (index_t i=0; i<m_multi_map_elements.get_num_elements(); ++i)
	{
		ParameterMapElement* current=m_multi_map_elements[i];
//		SG_SPRINT("element %d:\n", i)
//		SG_SPRINT("\tkey: ")
//		current->m_key->print_param_info();
//		SG_SPRINT("\t%d values:\n", current->m_values->get_num_elements())
		for (index_t j=0; j<current->m_values->get_num_elements(); ++j)
			current->m_values->get_element(j)->print_param_info("\t\t");
	}
}
