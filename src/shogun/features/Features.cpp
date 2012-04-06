/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Subset support written (W) 2011 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/features/Features.h>
#include <shogun/preprocessor/Preprocessor.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Parameter.h>

#include <string.h>

using namespace shogun;

CFeatures::CFeatures(int32_t size)
: CSGObject()
{
	init();
	cache_size = size;
}

CFeatures::CFeatures(const CFeatures& orig)
: CSGObject(orig)
{
	init();

	preproc = orig.preproc;
	num_preproc = orig.num_preproc;

	preprocessed=SG_MALLOC(bool, orig.num_preproc);
	memcpy(preprocessed, orig.preprocessed, sizeof(bool)*orig.num_preproc);
}

CFeatures::CFeatures(CFile* loader)
: CSGObject()
{
	init();

	load(loader);
	SG_INFO("Feature object loaded (%p)\n",this) ;
}

CFeatures::~CFeatures()
{
	clean_preprocessors();
	SG_UNREF(m_active_subset);
	SG_UNREF(m_subset_stack);
}

void
CFeatures::init()
{
	SG_ADD(&properties, "properties", "Feature properties", MS_NOT_AVAILABLE);
	SG_ADD(&cache_size, "cache_size", "Size of cache in MB", MS_NOT_AVAILABLE);

	/* TODO, use SGVector for arrays to be able to use SG_ADD macro */
	m_parameters->add_vector((CSGObject***) &preproc, &num_preproc, "preproc",
			"List of preprocessors");
	m_parameters->add_vector(&preprocessed, &num_preproc, "preprocessed",
			"Feature[i] is already preprocessed");

	SG_ADD((CSGObject**)&m_active_subset, "active_subset", "Subset object",
			MS_NOT_AVAILABLE);

	SG_ADD((CSGObject**)&m_subset_stack, "subset_stack",
			"Stack of subsets of indices", MS_NOT_AVAILABLE);

	m_subset_stack=new CList(true);
	SG_REF(m_subset_stack);

	m_active_subset=NULL;
	properties = FP_NONE;
	cache_size = 0;
	preproc = NULL;
	num_preproc = 0;
	preprocessed = NULL;
}

/// set preprocessor
int32_t CFeatures::add_preprocessor(CPreprocessor* p)
{
	SG_INFO( "%d preprocs currently, new preproc list is\n", num_preproc);
	ASSERT(p);

	bool* preprocd=SG_MALLOC(bool, num_preproc+1);
	CPreprocessor** pps=SG_MALLOC(CPreprocessor*, num_preproc+1);
	for (int32_t i=0; i<num_preproc; i++)
	{
		pps[i]=preproc[i];
		preprocd[i]=preprocessed[i];
	}
	SG_FREE(preproc);
	SG_FREE(preprocessed);
	preproc=pps;
	preprocessed=preprocd;
	preproc[num_preproc]=p;
	preprocessed[num_preproc]=false;

	num_preproc++;

	for (int32_t i=0; i<num_preproc; i++)
		SG_INFO( "preproc[%d]=%s %ld\n",i, preproc[i]->get_name(), preproc[i]) ;

	SG_REF(p);

	return num_preproc;
}

/// get current preprocessor
CPreprocessor* CFeatures::get_preprocessor(int32_t num)
{
	if (num<num_preproc)
	{
		SG_REF(preproc[num]);
		return preproc[num];
	}
	else
		return NULL;
}

/// get whether specified preprocessor (or all if num=1) was/were already applied
int32_t CFeatures::get_num_preprocessed()
{
	int32_t num=0;

	for (int32_t i=0; i<num_preproc; i++)
	{
		if (preprocessed[i])
			num++;
	}

	return num;
}

/// clears all preprocs
void CFeatures::clean_preprocessors()
{
	while (del_preprocessor(0));
}

/// del current preprocessor
CPreprocessor* CFeatures::del_preprocessor(int32_t num)
{
	CPreprocessor** pps=NULL;
	bool* preprocd=NULL;
	CPreprocessor* removed_preproc=NULL;

	if (num_preproc>0 && num<num_preproc)
	{
		removed_preproc=preproc[num];

		if (num_preproc>1)
		{
			pps= SG_MALLOC(CPreprocessor*, num_preproc-1);
			preprocd= SG_MALLOC(bool, num_preproc-1);

			if (pps && preprocd)
			{
				int32_t j=0;
				for (int32_t i=0; i<num_preproc; i++)
				{
					if (i!=num)
					{
						pps[j]=preproc[i];
						preprocd[j]=preprocessed[i];
						j++;
					}
				}
			}
		}

		SG_FREE(preproc);
		preproc=pps;
		SG_FREE(preprocessed);
		preprocessed=preprocd;

		num_preproc--;

		for (int32_t i=0; i<num_preproc; i++)
			SG_INFO( "preproc[%d]=%s\n",i, preproc[i]->get_name()) ;
	}

	SG_UNREF(removed_preproc);
	return removed_preproc;
}

void CFeatures::set_preprocessed(int32_t num)
{
	preprocessed[num]=true;
}

bool CFeatures::is_preprocessed(int32_t num)
{
	return preprocessed[num];
}

int32_t CFeatures::get_num_preprocessors() const
{
	return num_preproc;
}

int32_t CFeatures::get_cache_size()
{
	return cache_size;
}

bool CFeatures::reshape(int32_t num_features, int32_t num_vectors)
{
	SG_NOTIMPLEMENTED;
	return false;
}

void CFeatures::list_feature_obj()
{
	SG_INFO( "%p - ", this);
	switch (get_feature_class())
	{
		case C_UNKNOWN:
			SG_INFO( "C_UNKNOWN ");
			break;
		case C_SIMPLE:
			SG_INFO( "C_SIMPLE ");
			break;
		case C_SPARSE:
			SG_INFO( "C_SPARSE ");
			break;
		case C_STRING:
			SG_INFO( "C_STRING ");
			break;
		case C_COMBINED:
			SG_INFO( "C_COMBINED ");
			break;
		case C_COMBINED_DOT:
			SG_INFO( "C_COMBINED_DOT ");
			break;
		case C_WD:
			SG_INFO( "C_WD ");
			break;
		case C_SPEC:
			SG_INFO( "C_SPEC ");
			break;
		case C_WEIGHTEDSPEC:
			SG_INFO( "C_WEIGHTEDSPEC ");
			break;
		case C_STREAMING_SIMPLE:
			SG_INFO( "C_STREAMING_SIMPLE ");
			break;
		case C_STREAMING_SPARSE:
			SG_INFO( "C_STREAMING_SPARSE ");
			break;
		case C_STREAMING_STRING:
			SG_INFO( "C_STREAMING_STRING ");
			break;
		case C_STREAMING_VW:
			SG_INFO( "C_STREAMING_VW ");
			break;
		case C_ANY:
			SG_INFO( "C_ANY ");
			break;
		default:
         SG_ERROR( "ERROR UNKNOWN FEATURE CLASS");
	}

	switch (get_feature_type())
	{
		case F_UNKNOWN:
			SG_INFO( "F_UNKNOWN \n");
			break;
		case F_CHAR:
			SG_INFO( "F_CHAR \n");
			break;
		case F_BYTE:
			SG_INFO( "F_BYTE \n");
			break;
		case F_SHORT:
			SG_INFO( "F_SHORT \n");
			break;
		case F_WORD:
			SG_INFO( "F_WORD \n");
			break;
		case F_INT:
			SG_INFO( "F_INT \n");
			break;
		case F_UINT:
			SG_INFO( "F_UINT \n");
			break;
		case F_LONG:
			SG_INFO( "F_LONG \n");
			break;
		case F_ULONG:
			SG_INFO( "F_ULONG \n");
			break;
		case F_SHORTREAL:
			SG_INFO( "F_SHORTEAL \n");
			break;
		case F_DREAL:
			SG_INFO( "F_DREAL \n");
			break;
		case F_LONGREAL:
			SG_INFO( "F_LONGREAL \n");
			break;
		case F_ANY:
			SG_INFO( "F_ANY \n");
			break;
		default:
         SG_ERROR( "ERROR UNKNOWN FEATURE TYPE\n");
	}
}


void CFeatures::load(CFile* loader)
{
	SG_SET_LOCALE_C;
	SG_NOTIMPLEMENTED;
	SG_RESET_LOCALE;
}

void CFeatures::save(CFile* writer)
{
	SG_SET_LOCALE_C;
	SG_NOTIMPLEMENTED;
	SG_RESET_LOCALE;
}

bool CFeatures::check_feature_compatibility(CFeatures* f)
{
	bool result=false;

	if (f)
		result= ( (this->get_feature_class() == f->get_feature_class()) &&
				(this->get_feature_type() == f->get_feature_type()));
	return result;
}

bool CFeatures::has_property(EFeatureProperty p)
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

void CFeatures::push_subset(CSubset* subset)
{
	/* do some basic consistency checks */
	if (!subset)
		SG_ERROR("CFeatures::push_subset(NULL) is illegal.\n");

	/* check for legal size (only possible if there is already a subset) */
	if (has_subsets())
	{
		index_t available=m_active_subset->get_size();
		if (subset->get_size()>available)
			SG_ERROR("Pushed subset contains more indices than available.\n");

		if (subset->get_max_index()>= available)
			SG_ERROR("Pushed subset contains index out of bounds (too large).\n");
	}

	m_subset_stack->push(subset);
	update_active_subset();
	subset_changed_post();
}

bool CFeatures::has_subsets() const
{
	return m_subset_stack->get_num_elements();
}

void CFeatures::pop_subset()
{
	m_subset_stack->pop();
	update_active_subset();
	subset_changed_post();
}

void CFeatures::remove_all_subsets()
{
	m_subset_stack->delete_all_elements();
	update_active_subset();
	subset_changed_post();
}

void CFeatures::update_active_subset()
{
	/* delete active subset and rebuild from subset stack */
	SG_UNREF(m_active_subset);

	if (m_subset_stack->get_num_elements())
	{

		/* current_indices will contain the "real" indices which are translated
		 * iteratively through all stacked subsets. start with last subset */
		CSubset* current_subset=(CSubset*)m_subset_stack->get_last_element();
		SGVector<index_t> current_indices=SGVector<index_t>(
				current_subset->get_size());
		for (index_t i=0; i<current_indices.vlen; ++i)
			current_indices.vector[i]=current_subset->subset_idx_conversion(i);

		SG_UNREF(current_subset);
		current_subset=(CSubset*)m_subset_stack->get_previous_element();

		/* now remaining subsets */
		while(current_subset)
		{
			SGVector<index_t> new_indices=SGVector<index_t>(
					current_subset->get_size());

			/* translate current real indices through current subset */
			for (index_t i=0; i<current_indices.vlen; ++i)
			{
				new_indices.vector[i]=current_subset->subset_idx_conversion(
						current_indices.vector[i]);
			}

			/* replace current real indices */
			current_indices.destroy_vector();
			current_indices=SGVector<index_t>(new_indices);

			/* next subset */
			SG_UNREF(current_subset);
			current_subset=(CSubset*)m_subset_stack->get_next_element();
		}

		m_active_subset=new CSubset(current_indices);
	}
}

CFeatures* CFeatures::copy_subset(SGVector<index_t> indices)
{
	SG_ERROR("copy_subset and therefore model storage of CMachine "
			"(required for cross-validation and model-selection is ",
			"not yet implemented for feature type %s\n", get_name());
	return NULL;
}
