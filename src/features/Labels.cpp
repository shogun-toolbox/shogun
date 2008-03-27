/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/Labels.h"
#include "lib/common.h"
#include "lib/File.h"
#include "lib/io.h"
#include "lib/Mathematics.h"

CLabels::CLabels() : CSGObject()
{
	labels = NULL;
	num_labels = 0;
}

CLabels::CLabels(INT num_lab) : CSGObject(), num_labels(num_lab)
{
	labels=new DREAL[num_lab];
	ASSERT(labels);

	for (INT i=0; i<num_lab; i++)
		labels[i]=0;
}

CLabels::CLabels(CHAR* fname) : CSGObject()
{
	num_labels=0;
	labels=NULL;

	load(fname);
}

CLabels::~CLabels()
{
	delete[] labels;
	num_labels=0;
	labels=NULL;
}

void CLabels::set_labels(DREAL* p_labels, INT len)
{
	ASSERT(len>0);
	num_labels = len;

	delete[] labels;
	labels = new DREAL[len];
	ASSERT(labels);

	for (INT i=0; i<len; i++)
		labels[i] = p_labels[i];
}

bool CLabels::is_two_class_labeling()
{
	ASSERT(labels);

	for (INT i=0; i<num_labels; i++)
	{
		if (labels[i] != +1.0 && labels[i] != -1.0)
		{
			SG_ERROR("Not a two class labeling label[%d]=%f (only +1/-1 allowed)\n", i, labels[i]);
			return false;
		}
	}
	return true;
}

INT CLabels::get_num_classes()
{
	INT n=-1;
	INT* lab=get_int_labels(n);

	INT num_classes=0;
	for (INT i=0; i<n; i++)
		num_classes=CMath::max(num_classes,lab[i]);

	delete[] lab;

	return num_classes+1;
}

DREAL* CLabels::get_labels(INT &len)
{
	len=num_labels;

	if (num_labels>0)
	{
		DREAL* _labels=new DREAL[num_labels] ;
		for (INT i=0; i<len; i++)
			_labels[i]=get_label(i) ;
		return _labels ;
	}
	else 
		return NULL;
}

void CLabels::get_labels(DREAL** p_labels, INT* len)
{
	ASSERT(p_labels && len);
	*p_labels=NULL;
	*len=num_labels;

	if (num_labels>0)
	{
		*p_labels=(DREAL*) malloc(sizeof(DREAL)*num_labels);

		for (INT i=0; i<num_labels; i++)
			(*p_labels)[i]=get_label(i);
	}
}

INT* CLabels::get_int_labels(INT &len)
{
	len=num_labels;

	if (num_labels>0)
	{
		INT* _labels=new INT[num_labels] ;
		for (INT i=0; i<len; i++)
			_labels[i]= (INT) get_label(i) ;
		return _labels ;
	}
	else 
		return NULL;
}

void CLabels::set_int_labels(INT * mylabels, INT len)
{
	num_labels = len ;
	delete[] labels ;
	
	labels = new DREAL[num_labels] ;
	for (INT i=0; i<num_labels; i++)
		set_int_label(i, mylabels[i]) ;
}

bool CLabels::load(CHAR* fname)
{
	bool status=false;

	delete[] labels;
	num_labels=0;

	CFile f(fname, 'r', F_DREAL);
	LONG num_lab=0;
	labels=f.load_real_data(NULL, num_lab);
	num_labels=num_lab;

    if (!f.is_ok()) {
      SG_ERROR( "loading file \"%s\" failed", fname);
    }
	else
	{
		SG_INFO( "%ld labels successfully read\n", num_labels);
		status=true;
	}

	return status;
}

bool CLabels::save(CHAR* fname)
{
	return false;
}
