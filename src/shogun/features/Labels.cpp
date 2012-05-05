/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2011 Heiko Strathmann
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
}

void CLabels::set_to_one()
{
	ASSERT(labels.vector);
	index_t subset_size=get_num_labels();
	for (int32_t i=0; i<subset_size; i++)
		labels.vector[m_subset_stack->subset_idx_conversion(i)]=+1;
}

CLabels::~CLabels()
{
	SG_UNREF(m_subset_stack);
}

void CLabels::init()
{
	m_parameters->add(&labels, "labels", "The labels.");
	m_parameters->add((CSGObject**)&m_subset_stack, "subset_stack",
			"Current subset stack");

	labels=SGVector<float64_t>();
	m_subset_stack=new CSubsetStack();
	SG_REF(m_subset_stack);
}

void CLabels::set_labels(SGVector<float64_t> v)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("A subset is set, cannot set labels\n");

	SG_PRINT("before labels=v\n");
	labels.ref_count();
	v.ref_count();
	labels=v;
	labels.ref_count();
	v.ref_count();
	SG_PRINT("after labels=v\n");
}

bool CLabels::is_two_class_labeling()
{
	ASSERT(labels.vector);
	bool found_plus_one=false;
	bool found_minus_one=false;

	int32_t subset_size=get_num_labels();
	for (int32_t i=0; i<subset_size; i++)
	{
		int32_t real_i=m_subset_stack->subset_idx_conversion(i);
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
	SGVector<float64_t> unique=get_unique_labels();
	return unique.vlen;
}

SGVector<float64_t> CLabels::get_labels()
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("get_labels() is not possible on subset");

	return labels;
}

SGVector<float64_t> CLabels::get_labels_copy()
{
	if (!m_subset_stack->has_subsets())
		return labels.clone();

	index_t num_labels=get_num_labels();
	SGVector<float64_t> result(SG_MALLOC(float64_t, num_labels), num_labels);

	/* copy element wise because of possible subset */
	for (index_t i=0; i<num_labels; i++)
		result[i]=get_label(i);

	return result;
}

SGVector<float64_t> CLabels::get_unique_labels()
{
	/* extract all labels (copy because of possible subset) */
	SGVector<float64_t> unique_labels=get_labels_copy();
	unique_labels.vlen=CMath::unique(unique_labels.vector, unique_labels.vlen);

	SGVector<float64_t> result(unique_labels.vlen);
	memcpy(result.vector, unique_labels.vector,
			sizeof(float64_t)*unique_labels.vlen);

	return result;
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
	if (m_subset_stack->has_subsets())
		SG_ERROR("set_int_labels() is not possible on subset");

	labels = SGVector<float64_t>(lab.vlen);

	for (int32_t i=0; i<lab.vlen; i++)
		set_int_label(i, lab.vector[i]);
}

bool CLabels::set_label(int32_t idx, float64_t label)
{
	int32_t real_num=m_subset_stack->subset_idx_conversion(idx);
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
	int32_t real_num=m_subset_stack->subset_idx_conversion(idx);
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
	int32_t real_num=m_subset_stack->subset_idx_conversion(idx);
	ASSERT(labels.vector && idx<get_num_labels());
	return labels.vector[real_num];
}

int32_t CLabels::get_int_label(int32_t idx)
{
	int32_t real_num=m_subset_stack->subset_idx_conversion(idx);
	ASSERT(labels.vector && idx<get_num_labels());
	if (labels.vector[real_num] != float64_t((int32_t(labels.vector[real_num]))))
		SG_ERROR("label[%d]=%g is not an integer\n", labels.vector[real_num]);

	return int32_t(labels.vector[real_num]);
}

int32_t CLabels::get_num_labels()
{
	return m_subset_stack->has_subsets()
			? m_subset_stack->get_size() : labels.vlen;
}

void CLabels::add_subset(SGVector<index_t> subset)
{
	m_subset_stack->add_subset(subset);
}

void CLabels::remove_subset()
{
	m_subset_stack->remove_subset();
}

void CLabels::remove_all_subsets()
{
	m_subset_stack->remove_all_subsets();
}
