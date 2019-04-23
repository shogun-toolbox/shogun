/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg
 */

#include <shogun/features/Features.h>
#include <shogun/features/AttributeFeatures.h>
#include <shogun/lib/memory.h>

using namespace shogun;

AttributeFeatures::AttributeFeatures()
: Features(0)
{
}

std::shared_ptr<Features> AttributeFeatures::get_attribute(char* attr_name)
{
	int32_t idx=find_attr_index(attr_name);
	if (idx>=0)
	{
		return features[idx].attr_obj;
	}

	return NULL;
}

void AttributeFeatures::get_attribute_by_index(int idx, const char* &attr_name, std::shared_ptr<Features> &attr_obj)
{
		T_ATTRIBUTE a= features.get_element_safe(idx);
		attr_name= a.attr_name;
		attr_obj= a.attr_obj;
}

bool AttributeFeatures::set_attribute(char* attr_name, std::shared_ptr<Features> attr_obj)
{
	int32_t idx=find_attr_index(attr_name);
	if (idx==-1)
		idx=features.get_num_elements();

	T_ATTRIBUTE a;
	a.attr_name=get_strdup(attr_name);
	a.attr_obj=attr_obj;

	return features.set_element(a, idx);
}

bool AttributeFeatures::del_attribute(char* attr_name)
{
	int32_t idx=find_attr_index(attr_name);

	if (idx>=0)
	{
		T_ATTRIBUTE a= features[idx];
		SG_FREE(a.attr_name);
		a.attr_obj.reset();
		return true;
	}
	return false;
}

int32_t AttributeFeatures::get_num_attributes()
{
	return features.get_num_elements();
}

int32_t AttributeFeatures::find_attr_index(char* attr_name)
{
	int32_t n=features.get_num_elements();
	for (int32_t i=0; i<n; i++)
	{
		if (!strcmp(features[n].attr_name, attr_name))
			return i;
	}

	return -1;
}

AttributeFeatures::~AttributeFeatures()
{
	int32_t n=features.get_num_elements();
	for (int32_t i=0; i<n; i++)
		features[i].attr_obj.reset();
}
