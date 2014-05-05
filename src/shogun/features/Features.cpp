/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/features/Features.h>
#include <shogun/preprocessor/Preprocessor.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/DynamicObjectArray.h>

#include <string.h>

using namespace shogun;

CFeatures::CFeatures(int32_t size)
: CSGObject()
{
	init();
	cache_size = size;
}

CFeatures::CFeatures(CFile* loader)
: CSGObject()
{
	init();

	load(loader);
	SG_INFO("Feature object loaded (%p)\n",this)
}

CFeatures::~CFeatures()
{
	clean_preprocessors();
	SG_UNREF(m_subset_stack);
	SG_UNREF(preproc);
	SG_UNREF(preprocessed);
}

void CFeatures::init()
{
	SG_ADD(&properties, "properties", "Feature properties", MS_NOT_AVAILABLE);
	SG_ADD(&cache_size, "cache_size", "Size of cache in MB", MS_NOT_AVAILABLE);

	SG_ADD((CSGObject**) &preproc, "preproc", "Array of preprocessors.",
	       MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &preprocessed, "preprocessed", "Array of preprocessed.",
	       MS_NOT_AVAILABLE);

	SG_ADD((CSGObject**)&m_subset_stack, "subset_stack", "Stack of subsets",
	       MS_NOT_AVAILABLE);

	m_subset_stack=new CSubsetStack();
	SG_REF(m_subset_stack);

	properties = FP_NONE;
	cache_size = 0;
	preproc = new CDynamicObjectArray();
	preprocessed = new CDynamicArray<bool>();
	SG_REF(preproc);
	SG_REF(preprocessed);
}

void CFeatures::add_preprocessor(CPreprocessor* p)
{
	ASSERT(p)

	preproc->push_back(p);
	preprocessed->push_back(false);
}

CPreprocessor* CFeatures::get_preprocessor(int32_t num) const
{
	if (num<preproc->get_num_elements() && num>=0)
	{
	  return (CPreprocessor*) preproc->get_element(num);
	}
	else
		return NULL;
}

int32_t CFeatures::get_num_preprocessed() const
{
	int32_t num=0;

	for (int32_t i=0; i<preproc->get_num_elements(); i++)
	{
	  if ((*preprocessed)[i])
			num++;
	}

	return num;
}

void CFeatures::clean_preprocessors()
{
	preproc->reset_array();
	preprocessed->reset_array();
}

void CFeatures::del_preprocessor(int32_t num)
{
	if (num<preproc->get_num_elements() && num>=0)
	{
		preproc->delete_element(num);
		preprocessed->delete_element(num);
	}
}

void CFeatures::list_preprocessors()
{
	int32_t num_preproc = preproc->get_num_elements();

	for (int32_t i=0; i<num_preproc; i++)
	{
		SG_INFO("preproc[%d]=%s applied=%s\n",i,
				preproc->get_element(i)->get_name(),
				preprocessed->get_element(i) ? "true" : "false");
	}
}

void CFeatures::set_preprocessed(int32_t num)
{
	ASSERT(num<preprocessed->get_num_elements() && num>=0);
	(*preprocessed)[num]=true;
}

bool CFeatures::is_preprocessed(int32_t num) const
{
	ASSERT(num<preprocessed->get_num_elements() && num>=0);
	return (*preprocessed)[num];
}

int32_t CFeatures::get_num_preprocessors() const
{
	return preproc->get_num_elements();
}

int32_t CFeatures::get_cache_size() const
{
	return cache_size;
}

bool CFeatures::reshape(int32_t num_features, int32_t num_vectors)
{
	SG_NOTIMPLEMENTED
	return false;
}

void CFeatures::list_feature_obj() const
{
	SG_INFO("%p - ", this)
	switch (get_feature_class())
	{
		case C_UNKNOWN:
			SG_INFO("C_UNKNOWN ")
			break;
		case C_DENSE:
			SG_INFO("C_DENSE ")
			break;
		case C_SPARSE:
			SG_INFO("C_SPARSE ")
			break;
		case C_STRING:
			SG_INFO("C_STRING ")
			break;
		case C_COMBINED:
			SG_INFO("C_COMBINED ")
			break;
		case C_COMBINED_DOT:
			SG_INFO("C_COMBINED_DOT ")
			break;
		case C_WD:
			SG_INFO("C_WD ")
			break;
		case C_SPEC:
			SG_INFO("C_SPEC ")
			break;
		case C_WEIGHTEDSPEC:
			SG_INFO("C_WEIGHTEDSPEC ")
			break;
		case C_STREAMING_DENSE:
			SG_INFO("C_STREAMING_DENSE ")
			break;
		case C_STREAMING_SPARSE:
			SG_INFO("C_STREAMING_SPARSE ")
			break;
		case C_STREAMING_STRING:
			SG_INFO("C_STREAMING_STRING ")
			break;
		case C_STREAMING_VW:
			SG_INFO("C_STREAMING_VW ")
			break;
		case C_ANY:
			SG_INFO("C_ANY ")
			break;
		default:
         SG_ERROR("ERROR UNKNOWN FEATURE CLASS")
	}

	switch (get_feature_type())
	{
		case F_UNKNOWN:
			SG_INFO("F_UNKNOWN \n")
			break;
		case F_CHAR:
			SG_INFO("F_CHAR \n")
			break;
		case F_BYTE:
			SG_INFO("F_BYTE \n")
			break;
		case F_SHORT:
			SG_INFO("F_SHORT \n")
			break;
		case F_WORD:
			SG_INFO("F_WORD \n")
			break;
		case F_INT:
			SG_INFO("F_INT \n")
			break;
		case F_UINT:
			SG_INFO("F_UINT \n")
			break;
		case F_LONG:
			SG_INFO("F_LONG \n")
			break;
		case F_ULONG:
			SG_INFO("F_ULONG \n")
			break;
		case F_SHORTREAL:
			SG_INFO("F_SHORTEAL \n")
			break;
		case F_DREAL:
			SG_INFO("F_DREAL \n")
			break;
		case F_LONGREAL:
			SG_INFO("F_LONGREAL \n")
			break;
		case F_ANY:
			SG_INFO("F_ANY \n")
			break;
		default:
         SG_ERROR("ERROR UNKNOWN FEATURE TYPE\n")
	}
}


void CFeatures::load(CFile* loader)
{
	SG_SET_LOCALE_C;
	SG_NOTIMPLEMENTED
	SG_RESET_LOCALE;
}

void CFeatures::save(CFile* writer)
{
	SG_SET_LOCALE_C;
	SG_NOTIMPLEMENTED
	SG_RESET_LOCALE;
}

bool CFeatures::check_feature_compatibility(CFeatures* f) const
{
	bool result=false;

	if (f)
	{
		result= ( (this->get_feature_class() == f->get_feature_class()) &&
				(this->get_feature_type() == f->get_feature_type()));
	}
	return result;
}

bool CFeatures::has_property(EFeatureProperty p) const
{
	return (properties & p) != 0;
}

void CFeatures::set_property(EFeatureProperty p)
{
	properties |= p;
}

void CFeatures::unset_property(EFeatureProperty p)
{
	properties &= (properties | p) ^ p;
}

void CFeatures::add_subset(SGVector<index_t> subset)
{
	m_subset_stack->add_subset(subset);
	subset_changed_post();
}

void CFeatures::add_subset_in_place(SGVector<index_t> subset)
{
	m_subset_stack->add_subset_in_place(subset);
	subset_changed_post();
}

void CFeatures::remove_subset()
{
	m_subset_stack->remove_subset();
	subset_changed_post();
}

void CFeatures::remove_all_subsets()
{
	m_subset_stack->remove_all_subsets();
	subset_changed_post();
}

CSubsetStack* CFeatures::get_subset_stack()
{
	SG_REF(m_subset_stack);
	return m_subset_stack;
}

CFeatures* CFeatures::copy_subset(SGVector<index_t> indices)
{
	SG_ERROR("%s::copy_subset(): copy_subset and therefore model storage of "
			"CMachine (required for cross-validation and model-selection is "
			"not yet implemented yet. Ask developers!\n", get_name());
	return NULL;
}
