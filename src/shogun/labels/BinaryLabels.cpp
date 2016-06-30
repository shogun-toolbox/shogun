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

#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/lib/SGVector.h>

using namespace shogun;

CBinaryLabels::CBinaryLabels() : CDenseLabels()
{
}

CBinaryLabels::CBinaryLabels(int32_t num_labels) : CDenseLabels(num_labels)
{
}

#if !defined(SWIGJAVA) && !defined(SWIGCSHARP)
CBinaryLabels::CBinaryLabels(SGVector<int32_t> src) : CDenseLabels()
{
	SGVector<float64_t> values(src.vlen);
	for (int32_t i = 0; i < values.vlen; i++)
	{
		values[i] = src[i];
	}
	set_int_labels(src);
	set_values(values);
}

CBinaryLabels::CBinaryLabels(SGVector<int64_t> src) : CDenseLabels()
{
	SGVector<float64_t> values(src.vlen);
	for (int32_t i = 0; i < values.vlen; i++)
	{
		values[i] = src[i];
	}
	set_int_labels(src);
	set_values(values);
}
#endif

CBinaryLabels::CBinaryLabels(SGVector<float64_t> src, float64_t threshold) : CDenseLabels()
{
	SGVector<float64_t> labels(src.vlen);
	for (int32_t i = 0; i < labels.vlen; i++)
	{
		labels[i] = src[i] + threshold >= 0 ? +1.0 : -1.0;
	}
	set_labels(labels);
	set_values(src);
}

CBinaryLabels::CBinaryLabels(CFile * loader) : CDenseLabels(loader)
{
}

void CBinaryLabels::ensure_valid(const char * context)
{
	CDenseLabels::ensure_valid(context);
	bool found_plus_one = false;
	bool found_minus_one = false;

	int32_t subset_size = get_num_labels();
	for (int32_t i = 0; i < subset_size; i++)
	{
		int32_t real_i = m_subset_stack->subset_idx_conversion(i);
		if (m_labels[real_i] == +1.0)
		{
			found_plus_one = true;
		}
		else if (m_labels[real_i] == -1.0)
		{
			found_minus_one = true;
		}
		else
		{
			SG_ERROR(
			        "%s%s%s::ensure_valid(): Not a two class labeling label[%d]=%f (only +1/-1 "
			        "allowed)\n", context ? context : "",
			        context ? ": " : "", get_name(), i, m_labels[real_i]);
		}
	}

	if (!found_plus_one)
	{
		SG_WARNING(
		        "%s%s%s::ensure_valid(): Not a two class labeling - no positively labeled examples found\n",
		        context ? context : "", context ? ": " : "", get_name());
	}

	if (!found_minus_one)
	{
		SG_WARNING(
		        "%s%s%s::ensure_valid): Not a two class labeling - no negatively labeled examples found\n",
		        context ? context : "", context ? ": " : "", get_name());
	}
}

ELabelType CBinaryLabels::get_label_type() const
{
	return LT_BINARY;
}

void CBinaryLabels::scores_to_probabilities(float64_t a, float64_t b)
{
	SG_DEBUG("entering CBinaryLabels::scores_to_probabilities()\n")

	REQUIRE(m_current_values.vector, "%s::scores_to_probabilities() requires "
	        "values vector!\n", get_name());

	if (a == 0 && b == 0)
	{
		CStatistics::SigmoidParamters params =
		        CStatistics::fit_sigmoid(m_current_values);
		a = params.a;
		b = params.b;
	}

	SG_DEBUG("using sigmoid: a=%f, b=%f\n", a, b)

	/* now the sigmoid is fitted, convert all values to probabilities */
	for (index_t i = 0; i < m_current_values.vlen; ++i)
	{
		float64_t fApB = m_current_values[i] * a + b;
		m_current_values[i] = fApB >= 0 ? CMath::exp(-fApB) / (1.0 + CMath::exp(-fApB)) :
		                      1.0 / (1 + CMath::exp(fApB));
	}

	SG_DEBUG("leaving CBinaryLabels::scores_to_probabilities()\n")
}

CLabels* CBinaryLabels::shallow_subset_copy()
{
	CLabels* shallow_copy_labels=NULL;
	SGVector<float64_t> shallow_copy_vector(m_labels);
	shallow_copy_labels=new CBinaryLabels(m_labels.size());
	SG_REF(shallow_copy_labels);

	((CDenseLabels*) shallow_copy_labels)->set_labels(shallow_copy_vector);
	if (m_subset_stack->has_subsets())
		shallow_copy_labels->add_subset(m_subset_stack->get_last_subset()->get_subset_idx());

	return shallow_copy_labels;
}
