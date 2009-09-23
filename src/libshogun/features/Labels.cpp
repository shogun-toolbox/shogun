/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/Labels.h"
#include "lib/common.h"
#include "lib/File.h"
#include "lib/io.h"
#include "lib/Mathematics.h"

#ifdef HAVE_BOOST_SERIALIZATION
#include <boost/serialization/export.hpp>
BOOST_CLASS_EXPORT(CLabels);
#endif //HAVE_BOOST_SERIALIZATION

CLabels::CLabels()
: CSGObject()
{
	labels = NULL;
	num_labels = 0;
}

CLabels::CLabels(int32_t num_lab)
: CSGObject(), num_labels(num_lab)
{
	labels=new float64_t[num_lab];
	for (int32_t i=0; i<num_lab; i++)
		labels[i]=0;
}

CLabels::CLabels(float64_t* p_labels, int32_t len)
: CSGObject()
{
	labels = NULL;
	num_labels = 0;

    set_labels(p_labels, len);
}

CLabels::CLabels(char* fname)
: CSGObject()
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

void CLabels::set_labels(float64_t* p_labels, int32_t len)
{
	ASSERT(len>0);
	num_labels=len;

	delete[] labels;
    labels=CMath::clone_vector(p_labels, len);
}

bool CLabels::is_two_class_labeling()
{
	ASSERT(labels);
	bool found_plus_one=false;
	bool found_minus_one=false;

	for (int32_t i=0; i<num_labels; i++)
	{
		if (labels[i]==+1.0)
			found_plus_one=true;
		else if (labels[i]==-1.0)
			found_minus_one=true;
		else
			SG_ERROR("Not a two class labeling label[%d]=%f (only +1/-1 allowed)\n", i, labels[i]);
	}

	if (!found_plus_one)
		SG_ERROR("Not a two class labeling - no positively labeled examples found\n");
	if (!found_minus_one)
		SG_ERROR("Not a two class labeling - no negatively labeled examples found\n");

	return true;
}

int32_t CLabels::get_num_classes()
{
	int32_t n=-1;
	int32_t* lab=get_int_labels(n);

	int32_t num_classes=0;
	for (int32_t i=0; i<n; i++)
		num_classes=CMath::max(num_classes,lab[i]);

	delete[] lab;

	return num_classes+1;
}

float64_t* CLabels::get_labels(int32_t &len)
{
	len=num_labels;

	if (num_labels>0)
	{
		float64_t* _labels=new float64_t[num_labels] ;
		for (int32_t i=0; i<len; i++)
			_labels[i]=get_label(i) ;
		return _labels ;
	}
	else 
		return NULL;
}

void CLabels::get_labels(float64_t** p_labels, int32_t* len)
{
	ASSERT(p_labels && len);
	*p_labels=NULL;
	*len=num_labels;

	if (num_labels>0)
	{
		*p_labels=(float64_t*) malloc(sizeof(float64_t)*num_labels);

		for (int32_t i=0; i<num_labels; i++)
			(*p_labels)[i]=get_label(i);
	}
}

int32_t* CLabels::get_int_labels(int32_t &len)
{
	len=num_labels;

	if (num_labels>0)
	{
		int32_t* _labels=new int32_t[num_labels] ;
		for (int32_t i=0; i<len; i++)
			_labels[i]= (int32_t) get_label(i) ;
		return _labels ;
	}
	else 
		return NULL;
}

void CLabels::set_int_labels(int32_t * mylabels, int32_t len)
{
	num_labels = len ;
	delete[] labels ;
	
	labels = new float64_t[num_labels] ;
	for (int32_t i=0; i<num_labels; i++)
		set_int_label(i, mylabels[i]) ;
}

bool CLabels::load(char* fname)
{
	bool status=false;

	delete[] labels;
	num_labels=0;

	CFile f(fname, 'r', F_DREAL);
	int64_t num_lab=0;
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

bool CLabels::save(char* fname)
{
	return false;
}
