/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Saurabh Mahindre, Sergey Lisitsyn,
 *          Soeren Sonnenburg, Fernando Iglesias, Evgeniy Andreev,
 *          Chiyuan Zhang, Olivier NGuyen, Thoralf Klein
 */

#include <shogun/base/range.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/DenseLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Statistics.h>

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
		labels[i] = src[i] >= threshold ? +1.0 : -1.0;
	}
	set_labels(labels);
	set_values(src);
}

CBinaryLabels::CBinaryLabels(CFile * loader) : CDenseLabels(loader)
{
}

bool CBinaryLabels::is_valid() const
{
	if (!CDenseLabels::is_valid())
		return false;

	int32_t subset_size = get_num_labels();
	for (int32_t i = 0; i < subset_size; i++)
	{
		int32_t real_i = m_subset_stack->subset_idx_conversion(i);
		if (m_labels[real_i] != +1.0 && m_labels[real_i] != -1.0)
			return false;
	}
	return true;
}

void CBinaryLabels::ensure_valid(const char* context)
{
	REQUIRE(is_valid(), "Binary Labels must be -1 or +1!\n");
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
		m_current_values[i] = fApB >= 0
		                          ? std::exp(-fApB) / (1.0 + std::exp(-fApB))
		                          : 1.0 / (1 + std::exp(fApB));
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

CBinaryLabels::CBinaryLabels(const CDenseLabels& dense) : CDenseLabels(dense)
{
	ensure_valid();
}

namespace shogun
{
	Some<CBinaryLabels> binary_labels(CLabels* orig)
	{
		REQUIRE(orig, "No labels provided.\n");
		try
		{
			switch (orig->get_label_type())
			{
			case LT_BINARY:
				return Some<CBinaryLabels>::from_raw((CBinaryLabels*)orig);
			default:
				SG_SNOTIMPLEMENTED
			}
		}
		catch (const ShogunException& e)
		{
			SG_SERROR(
			    "Cannot convert %s to binary labels: %s\n", orig->get_name(),
			    e.what());
		}

		return Some<CBinaryLabels>::from_raw(nullptr);
	}
} // namespace shogun
