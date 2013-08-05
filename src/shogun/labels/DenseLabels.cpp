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

#include <shogun/labels/Labels.h>
#include <shogun/labels/DenseLabels.h>
#include <shogun/lib/common.h>
#include <shogun/io/File.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

CDenseLabels::CDenseLabels()
: CLabels()
{
	init();
}

CDenseLabels::CDenseLabels(int32_t num_lab)
: CLabels()
{
	init();
	m_labels = SGVector<float64_t>(num_lab);
	m_current_values=SGVector<float64_t>(num_lab);
}

CDenseLabels::CDenseLabels(CFile* loader)
: CLabels()
{
	init();
	load(loader);
}

CDenseLabels::~CDenseLabels()
{
}

void CDenseLabels::init()
{
	SG_ADD(&m_labels, "labels", "The labels.", MS_NOT_AVAILABLE);
}

void CDenseLabels::set_to_one()
{
	set_to_const(1.0);
}

void CDenseLabels::zero()
{
	set_to_const(0.0);
}

void CDenseLabels::set_to_const(float64_t c)
{
	ASSERT(m_labels.vector)
	index_t subset_size=get_num_labels();
	for (int32_t i=0; i<subset_size; i++)
	{
		m_labels.vector[m_subset_stack->subset_idx_conversion(i)]=c;
		m_current_values.vector[m_subset_stack->subset_idx_conversion(i)]=c;
	}
}

void CDenseLabels::set_labels(SGVector<float64_t> v)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("A subset is set, cannot set labels\n")

	m_labels = v;
}

SGVector<float64_t> CDenseLabels::get_labels()
{
	if (m_subset_stack->has_subsets())
		return get_labels_copy();

	return m_labels;
}

SGVector<float64_t> CDenseLabels::get_labels_copy()
{
	if (!m_subset_stack->has_subsets())
		return m_labels.clone();

	index_t num_labels = get_num_labels();
	SGVector<float64_t> result(num_labels);

	/* copy element wise because of possible subset */
	for (index_t i=0; i<num_labels; i++)
		result[i] = get_label(i);

	return result;
}

SGVector<int32_t> CDenseLabels::get_int_labels()
{
	SGVector<int32_t> intlab(get_num_labels());

	for (int32_t i=0; i<get_num_labels(); i++)
		intlab.vector[i] = get_int_label(i);

	return intlab;
}

void CDenseLabels::set_int_labels(SGVector<int32_t> lab)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("set_int_labels() is not possible on subset")

	m_labels = SGVector<float64_t>(lab.vlen);

	for (int32_t i=0; i<lab.vlen; i++)
		set_int_label(i, lab.vector[i]);
}

#if !defined(SWIGJAVA) && !defined(SWIGCSHARP)
void CDenseLabels::set_int_labels(SGVector<int64_t> lab)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("set_int_labels() is not possible on subset")

	m_labels = SGVector<float64_t>(lab.vlen);

	for (int32_t i=0; i<lab.vlen; i++)
		set_int_label(i, lab.vector[i]);
}
#endif

void CDenseLabels::ensure_valid(const char* context)
{
    if (m_labels.vector == NULL)
        SG_ERROR("%s%sempty content (NULL) for labels\n", context?context:"", context?": ":"")
}

void CDenseLabels::load(CFile* loader)
{
	remove_subset();
	m_labels = SGVector<float64_t>();
	m_labels.load(loader);
}

void CDenseLabels::save(CFile* writer)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("save() is not possible on subset")

	m_labels.save(writer);
}

bool CDenseLabels::set_label(int32_t idx, float64_t label)
{
	int32_t real_num=m_subset_stack->subset_idx_conversion(idx);
	if (m_labels.vector && real_num<get_num_labels())
	{
		m_labels.vector[real_num]=label;
		return true;
	}
	else
		return false;
}

bool CDenseLabels::set_int_label(int32_t idx, int32_t label)
{
	int32_t real_num=m_subset_stack->subset_idx_conversion(idx);
	if (m_labels.vector && real_num<get_num_labels())
	{
		m_labels.vector[real_num] = (float64_t)label;
		return true;
	}
	else
		return false;
}

float64_t CDenseLabels::get_label(int32_t idx)
{
	int32_t real_num=m_subset_stack->subset_idx_conversion(idx);
	ASSERT(m_labels.vector && idx<get_num_labels())
	return m_labels.vector[real_num];
}

int32_t CDenseLabels::get_int_label(int32_t idx)
{
	int32_t real_num=m_subset_stack->subset_idx_conversion(idx);
	ASSERT(m_labels.vector && idx<get_num_labels())
	if (m_labels.vector[real_num] != float64_t((int32_t(m_labels.vector[real_num]))))
		SG_ERROR("label[%d]=%g is not an integer\n", m_labels.vector[real_num])

	return int32_t(m_labels.vector[real_num]);
}

int32_t CDenseLabels::get_num_labels()
{
	return m_subset_stack->has_subsets()
			? m_subset_stack->get_size() : m_labels.vlen;
}
