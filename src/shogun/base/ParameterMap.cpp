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
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

SGParamInfo::SGParamInfo()
{
	init();
}

SGParamInfo::SGParamInfo(const SGParamInfo& orig)
{
	init();

	/* copy name */
	m_name=strdup(orig.m_name);

	m_ctype=orig.m_ctype;
	m_stype=orig.m_stype;
	m_ptype=orig.m_ptype;
	m_param_version=orig.m_param_version;
}

SGParamInfo::SGParamInfo(const char* name, EContainerType ctype,
		EStructType stype, EPrimitiveType ptype, int32_t param_version)
{
	init();

	/* copy name */
	m_name=strdup(name);

	m_ctype=ctype;
	m_stype=stype;
	m_ptype=ptype;
	m_param_version=param_version;
}

SGParamInfo::SGParamInfo(const TParameter* param, int32_t param_version)
{
	init();

	/* copy name */
	m_name=strdup(param->m_name);

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

void SGParamInfo::print_param_info() const
{
	char* s=to_string();
	SG_SPRINT("%s\n", s);
	SG_FREE(s);
}

SGParamInfo* SGParamInfo::duplicate() const
{
	return new SGParamInfo(m_name, m_ctype, m_stype, m_ptype, m_param_version);
}

void SGParamInfo::init()
{
	m_name=NULL;
	m_ctype=(EContainerType) -1;
	m_stype=(EStructType) -1;
	m_ptype=(EPrimitiveType) -1;
	m_param_version=-1;
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
	return m_ctype<0 && m_stype<0 && m_ptype<0 && !m_name;
}

ParameterMapElement::ParameterMapElement()
{
	init();
}

ParameterMapElement::ParameterMapElement(const SGParamInfo* key,
		const SGParamInfo* value)
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

void ParameterMap::put(const SGParamInfo* key, const SGParamInfo* value)
{
//	if (key->m_ptype==PT_SGOBJECT || value->m_ptype==PT_SGOBJECT)
//	{
//		SG_SPRINT("Parameter maps for CSGObjects are not yet supported\n");
//		SG_SNOTIMPLEMENTED;
//	}

	/* assert that versions do differ exactly one if mapping is non-empty */
	if(key->m_param_version-value->m_param_version!=1)
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

	m_map_elements.append_element(new ParameterMapElement(key, value));
	m_finalized=false;
}

const SGParamInfo* ParameterMap::get(const SGParamInfo* key) const
{
	index_t num_elements=m_map_elements.get_num_elements();

	/* check if underlying array is sorted */
	if (!m_finalized && num_elements>1)
		SG_SERROR("Call finalize_map() before calling get()\n");

	/* do binary search in array of pointers */
	/* dummy element for searching */
	ParameterMapElement* dummy=new ParameterMapElement(key->duplicate(),
			key->duplicate());
	index_t index=CMath::binary_search<ParameterMapElement> (
			m_map_elements.get_array(), num_elements, dummy);
	delete dummy;

	if (index==-1)
		return NULL;

	ParameterMapElement* element=m_map_elements.get_element(index);
	const SGParamInfo* value=element->m_value;

	return value;
}

void ParameterMap::finalize_map()
{
	/* sort underlying array */
	CMath::qsort<ParameterMapElement> (m_map_elements.get_array(),
			m_map_elements.get_num_elements());

	/* Now for every kel elementy, create a set of value elements.
	 * This set possibly contains more than one element if there were multiple
	 * values for one key. This is not a map in the common sense anymore,
	 * hoever, it suits the purpose here */
	//TODO
	/* create a hidden second structure map elements which is based on SGParamInfo
	 * keys and DynArray<const SGParamInfo*> as value. get then operates on this
	 * thing returnung a set of param infos
	 */

	/* ensure that there are no duplicate key->value pairs */
	//TODO
//	if (m_map_elements.get_num_elements()>1)
//	{
//		for (index_t i=1; i<m_map_elements.get_num_elements(); ++i)
//		{
//			const SGParamInfo* key1=m_map_elements.get_element(i-1)->m_key;
//			const SGParamInfo* key2=m_map_elements.get_element(i)->m_key;
//			if (*key1==*key2)
//				SG_SERROR("ParameterMap::finalize_map(): duplicate key!\n");
//		}
//	}


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
