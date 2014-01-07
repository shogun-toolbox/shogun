/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <features/Features.h>
#include <features/AttributeFeatures.h>
#include <lib/memory.h>

using namespace shogun;

CAttributeFeatures::CAttributeFeatures()
: CFeatures(0)
{
}

CFeatures* CAttributeFeatures::get_attribute(char* attr_name)
{
	int32_t idx=find_attr_index(attr_name);
	if (idx>=0)
	{
		CFeatures* f=features[idx].attr_obj;
		SG_REF(f);
		return f;
	}

	return NULL;
}

void CAttributeFeatures::get_attribute_by_index(int idx, const char* &attr_name, CFeatures* &attr_obj)
{
		T_ATTRIBUTE a= features.get_element_safe(idx);
		attr_name= a.attr_name;
		attr_obj= a.attr_obj;
		SG_REF(a.attr_obj);
}

bool CAttributeFeatures::set_attribute(char* attr_name, CFeatures* attr_obj)
{
	int32_t idx=find_attr_index(attr_name);
	if (idx==-1)
		idx=features.get_num_elements();

	T_ATTRIBUTE a;
	a.attr_name=get_strdup(attr_name);
	a.attr_obj=attr_obj;

	SG_REF(attr_obj);

	return features.set_element(a, idx);
}

bool CAttributeFeatures::del_attribute(char* attr_name)
{
	int32_t idx=find_attr_index(attr_name);

	if (idx>=0)
	{
		T_ATTRIBUTE a= features[idx];
		SG_FREE(a.attr_name);
		SG_UNREF(a.attr_obj);
		return true;
	}
	return false;
}

int32_t CAttributeFeatures::get_num_attributes()
{
	return features.get_num_elements();
}

int32_t CAttributeFeatures::find_attr_index(char* attr_name)
{
	int32_t n=features.get_num_elements();
	for (int32_t i=0; i<n; i++)
	{
		if (!strcmp(features[n].attr_name, attr_name))
			return i;
	}

	return -1;
}

CAttributeFeatures::~CAttributeFeatures()
{
	int32_t n=features.get_num_elements();
	for (int32_t i=0; i<n; i++)
		SG_UNREF_NO_NULL(features[i].attr_obj);
}
