/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/Features.h"
#include "preproc/PreProc.h"
#include "lib/io.h"

#include <string.h>

CFeatures::CFeatures(INT size) : cache_size(size), preproc(NULL), num_preproc(0), preprocessed(NULL) 
{
	CIO::message(M_INFO, "Feature object created (%ld)\n",this);
}

CFeatures::CFeatures(const CFeatures& orig) : preproc(orig.preproc), num_preproc(orig.num_preproc), preprocessed(orig.preprocessed)
{
	preprocessed=new bool[orig.num_preproc];
	ASSERT(preprocessed);
	memcpy(preprocessed, orig.preprocessed, sizeof(bool)*orig.num_preproc);
}

CFeatures::CFeatures(CHAR* fname) : cache_size(0), preproc(NULL), num_preproc(0), preprocessed(false)
{
	load(fname);
	CIO::message(M_INFO, "Feature object loaded (%ld)\n",this) ;
}

CFeatures::~CFeatures()
{
	CIO::message(M_INFO, "Feature object destroyed (%ld)\n",this) ;
}

/// set preprocessor
INT CFeatures::add_preproc(CPreProc* p)
{ 
	CIO::message(M_INFO, "%d preprocs currently, new preproc list is\n", num_preproc);
	INT i;

	bool* preprocd=new bool[num_preproc+1];
	CPreProc** pps=new CPreProc*[num_preproc+1];
	for (i=0; i<num_preproc; i++)
	{
		pps[i]=preproc[i];
		preprocd[i]=preprocessed[i];
	}
	delete[] preproc;
	delete[] preprocessed;
	preproc=pps;
	preprocessed=preprocd;
	preproc[num_preproc]=p;
	preprocessed[num_preproc]=false;

	num_preproc++;

	for (i=0; i<num_preproc; i++)
		CIO::message(M_INFO, "preproc[%d]=%s %ld\n",i, preproc[i]->get_name(), preproc[i]) ;
	return num_preproc;
}

/// get current preprocessor
CPreProc* CFeatures::get_preproc(INT num)
{ 
	if (num<num_preproc)
		return preproc[num];
	else
		return NULL;
}

/// get whether specified preprocessor (or all if num=1) was/were already applied
INT CFeatures::get_num_preprocessed()
{
	INT num=0;

	for (INT i=0; i<num_preproc; i++)
	{
		if (preprocessed[i])
			num++;
	}

	return num;
}

/// clears all preprocs
void CFeatures::clean_preprocs()
{
	while (del_preproc(0));
}

/// del current preprocessor
CPreProc* CFeatures::del_preproc(INT num)
{
	CPreProc** pps=NULL; 
	bool* preprocd=NULL; 
	CPreProc* removed_preproc=NULL;

	if (num_preproc>0 && num<num_preproc)
	{
		removed_preproc=preproc[num];

		if (num_preproc>1)
		{
			pps= new CPreProc*[num_preproc-1];
			preprocd= new bool[num_preproc-1];

			if (pps && preprocd)
			{
				INT j=0;
				for (INT i=0; i<num_preproc; i++)
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

		delete[] preproc;
		preproc=pps;
		delete[] preprocessed;
		preprocessed=preprocd;

		num_preproc--;

		for (INT i=0; i<num_preproc; i++)
			CIO::message(M_INFO, "preproc[%d]=%s\n",i, preproc[i]->get_name()) ;
	}

	return removed_preproc;
}

void CFeatures::list_feature_obj()
{
	CIO::message(M_INFO, "0x%X - ", this);
	switch (get_feature_class())
	{
		case C_UNKNOWN:
			CIO::message(M_INFO, "C_UNKNOWN ");
			break;
		case C_SIMPLE:
			CIO::message(M_INFO, "C_SIMPLE ");
			break;
		case C_SPARSE:
			CIO::message(M_INFO, "C_SPARSE ");
			break;
		case C_STRING:
			CIO::message(M_INFO, "C_STRING ");
			break;
		case C_COMBINED:
			CIO::message(M_INFO, "C_COMBINED ");
			break;
		case C_ANY:
			CIO::message(M_INFO, "C_ANY ");
			break;
		default:
#ifdef HAVE_PYTHON
         throw FeatureException("ERROR UNKNOWN FEATURE CLASS");
#else
			CIO::message(M_INFO, "ERROR UNKNOWN FEATURE CLASS");
#endif
	}

	switch (get_feature_type())
	{
		case F_UNKNOWN:
			CIO::message(M_INFO, "F_UNKNOWN \n");
			break;
		case F_DREAL:
			CIO::message(M_INFO, "F_REAL \n");
			break;
		case F_SHORT:
			CIO::message(M_INFO, "F_SHORT \n");
			break;
		case F_CHAR:
			CIO::message(M_INFO, "F_CHAR \n");
			break;
		case F_INT:
			CIO::message(M_INFO, "F_INT \n");
			break;
		case F_BYTE:
			CIO::message(M_INFO, "F_BYTE \n");
			break;
		case F_WORD:
			CIO::message(M_INFO, "F_WORD \n");
			break;
		case F_ULONG:
			CIO::message(M_INFO, "F_ULONG ");
			break;
		case F_ANY:
			CIO::message(M_INFO, "F_ANY \n");
			break;
		default:
#ifdef HAVE_PYTHON
         throw FeatureException("ERROR UNKNOWN FEATURE TYPE\n");
#else
			CIO::message(M_INFO, "ERROR UNKNOWN FEATURE TYPE\n");
#endif
	}
}

bool CFeatures::load(CHAR* fname)
{
	return false;
}

bool CFeatures::save(CHAR* fname)
{
	return false;
}

bool CFeatures::check_feature_compatibility(CFeatures* f)
{
	bool result=false;

	if (f)
		result= ( (this->get_feature_class() == f->get_feature_class()) &&
				(this->get_feature_type() == f->get_feature_type()));
	return result;
}
