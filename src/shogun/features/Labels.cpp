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

#include <shogun/features/Labels.h>
#include <shogun/lib/common.h>
#include <shogun/io/File.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/Set.h>

using namespace shogun;

CLabels::CLabels()
: CSGObject()
{
	init();
}

CLabels::CLabels(int32_t num_lab)
: CSGObject()
{
	init();
	labels=SGVector<float64_t>(num_lab);
}

CLabels::CLabels(SGVector<float64_t> src)
: CSGObject()
{
	init();

	set_labels(src);
	m_num_classes=get_num_classes();
}

CLabels::CLabels(SGVector<int64_t> src)
: CSGObject()
{
	init();

	SGVector<float64_t> converted(src.vlen);	
	for (int32_t i=0; i<src.vlen; i++)
		converted[i] = (float64_t)src[i];
	src.free_vector();		

	set_labels(converted);
	m_num_classes=get_num_classes();
}

void CLabels::set_to_one()
{
	ASSERT(labels.vector);
	index_t subset_size=get_num_labels();
	for (int32_t i=0; i<subset_size; i++)
		labels.vector[subset_idx_conversion(i)]=+1;
}

CLabels::CLabels(CFile* loader)
: CSGObject()
{
	init();
	load(loader);
}

CLabels::~CLabels()
{
	labels.destroy_vector();
	delete m_subset;
	m_subset=NULL;

	m_num_classes=0;
}

void CLabels::init()
{
	m_parameters->add(&labels, "labels", "The labels.");
	m_parameters->add((CSGObject**)&m_subset, "subset", "Subset object");

	labels=SGVector<float64_t>();
	m_num_classes=0;
	m_subset=NULL;
}

void CLabels::set_labels(SGVector<float64_t> v)
{
	if (m_subset)
		SG_ERROR("A subset is set, cannot set labels\n");

	labels.free_vector();
	labels=v;
	labels.do_free=false;
}

bool CLabels::is_two_class_labeling()
{
	ASSERT(labels.vector);
	bool found_plus_one=false;
	bool found_minus_one=false;

	int32_t subset_size=get_num_labels();
	for (int32_t i=0; i<subset_size; i++)
	{
		int32_t real_i=subset_idx_conversion(i);
		if (labels.vector[real_i]==+1.0)
			found_plus_one=true;
		else if (labels.vector[real_i]==-1.0)
			found_minus_one=true;
		else
		{
			SG_ERROR("Not a two class labeling label[%d]=%f (only +1/-1 "
					"allowed)\n", i, labels.vector[real_i]);
		}
	}

	if (!found_plus_one)
		SG_ERROR("Not a two class labeling - no positively labeled examples found\n");
	if (!found_minus_one)
		SG_ERROR("Not a two class labeling - no negatively labeled examples found\n");

	return true;
}

int32_t CLabels::get_num_classes()
{
	CSet<float64_t>* classes=new CSet<float64_t>();
	for (int32_t i=0; i<get_num_labels(); i++)
		classes->add(get_label(i));

	int32_t result=classes->get_num_elements();
	SG_UNREF(classes);
	return result;
}

SGVector<float64_t> CLabels::get_classes()
{
	CSet<float64_t>* classes=new CSet<float64_t>();

	for (int32_t i=0; i<get_num_labels(); i++)
		classes->add(get_label(i));

	SGVector<float64_t> result(classes->get_num_elements());
	memcpy(result.vector, classes->get_array(),
			sizeof(float64_t)*classes->get_num_elements());

	SG_UNREF(classes);
	return result;
}

SGVector<float64_t> CLabels::get_labels()
{
	if (m_subset)
		SG_ERROR("get_labels() is not possible on subset");

	return labels;
}

SGVector<int32_t> CLabels::get_int_labels()
{
	SGVector<int32_t> intlab(get_num_labels(), true);

	for (int32_t i=0; i<get_num_labels(); i++)
		intlab.vector[i]= get_int_label(i);

	return intlab;
}

void CLabels::set_int_labels(SGVector<int32_t> lab)
{
	if (m_subset)
		SG_ERROR("set_int_labels() is not possible on subset");

	labels.free_vector();
	labels = SGVector<float64_t>(lab.vlen);

	for (int32_t i=0; i<lab.vlen; i++)
		set_int_label(i, labels.vector[i]);
}

void CLabels::load(CFile* loader)
{
	remove_subset();

	SG_SET_LOCALE_C;
	labels.free_vector();

	ASSERT(loader);
	loader->get_vector(labels.vector, labels.vlen);
	m_num_classes=get_num_classes();
	SG_RESET_LOCALE;
}

void CLabels::save(CFile* writer)
{
	if (m_subset)
		SG_ERROR("save() is not possible on subset");

	SG_SET_LOCALE_C;
	ASSERT(writer);
	ASSERT(labels.vector && labels.vlen>0);
	writer->set_vector(labels.vector, labels.vlen);
	SG_RESET_LOCALE;
}

bool CLabels::set_label(int32_t idx, float64_t label)
{
	int32_t real_num=subset_idx_conversion(idx);
	if (labels.vector && real_num<get_num_labels())
	{
		labels.vector[real_num]=label;
		return true;
	}
	else 
		return false;
}

bool CLabels::set_int_label(int32_t idx, int32_t label)
{ 
	int32_t real_num=subset_idx_conversion(idx);
	if (labels.vector && real_num<get_num_labels())
	{
		labels.vector[real_num]= (float64_t) label;
		return true;
	}
	else 
		return false;
}

float64_t CLabels::get_label(int32_t idx)
{
	int32_t real_num=subset_idx_conversion(idx);
	ASSERT(labels.vector && idx<get_num_labels());
	return labels.vector[real_num];
}

int32_t CLabels::get_int_label(int32_t idx)
{
	int32_t real_num=subset_idx_conversion(idx);
	ASSERT(labels.vector && idx<get_num_labels());
	if (labels.vector[real_num] != float64_t((int32_t(labels.vector[real_num]))))
		SG_ERROR("label[%d]=%g is not an integer\n", labels.vector[real_num]);

	return int32_t(labels.vector[real_num]);
}

int32_t CLabels::get_num_labels()
{
	return m_subset ? m_subset->get_size() : labels.vlen;
}

void CLabels::set_subset(CSubset* subset)
{
	SG_UNREF(m_subset);
	m_subset=subset;
	SG_REF(subset);
}

void CLabels::remove_subset()
{
	set_subset(NULL);
}

index_t CLabels::subset_idx_conversion(index_t idx) const
{
	return m_subset ? m_subset->subset_idx_conversion(idx) : idx;
}
