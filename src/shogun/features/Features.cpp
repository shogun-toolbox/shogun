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

	preprocessed=new bool[orig.num_preproc];
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
	delete m_subset;
}

void
CFeatures::init(void)
{
	m_parameters->add(&properties, "properties",
					  "Feature properties.");
	m_parameters->add(&cache_size, "cache_size",
					  "Size of cache in MB.");

	m_parameters->add_vector((CSGObject***) &preproc,
							 &num_preproc, "preproc",
							 "List of preprocessors.");
	m_parameters->add_vector(&preprocessed,
							 &num_preproc, "preprocessed",
							 "Feature[i] is already preprocessed.");

	m_parameters->add((CSGObject**)&m_subset, "subset", "Subset object");

	m_subset=NULL;
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

	bool* preprocd=new bool[num_preproc+1];
	CPreprocessor** pps=new CPreprocessor*[num_preproc+1];
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
			pps= new CPreprocessor*[num_preproc-1];
			preprocd= new bool[num_preproc-1];

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

bool CFeatures::check_feature_compatibility(CFeatures* f)
{
	bool result=false;

	if (f)
		result= ( (this->get_feature_class() == f->get_feature_class()) &&
				(this->get_feature_type() == f->get_feature_type()));
	return result;
}

void CFeatures::set_subset(CSubset* subset)
{
	SG_UNREF(m_subset);
	m_subset=subset;
	SG_REF(subset);
	subset_changed_post();
}

void CFeatures::remove_subset()
{
	set_subset(NULL);
}
