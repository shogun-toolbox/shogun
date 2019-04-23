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

BinaryLabels::BinaryLabels() : DenseLabels()
{
}

BinaryLabels::BinaryLabels(int32_t num_labels) : DenseLabels(num_labels)
{
}

#if !defined(SWIGJAVA) && !defined(SWIGCSHARP)
BinaryLabels::BinaryLabels(SGVector<int32_t> src) : DenseLabels()
{
	SGVector<float64_t> values(src.vlen);
	for (int32_t i = 0; i < values.vlen; i++)
	{
		values[i] = src[i];
	}
	set_int_labels(src);
	set_values(values);
}

BinaryLabels::BinaryLabels(SGVector<int64_t> src) : DenseLabels()
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

BinaryLabels::BinaryLabels(SGVector<float64_t> src, float64_t threshold) : DenseLabels()
{
	SGVector<float64_t> labels(src.vlen);
	for (int32_t i = 0; i < labels.vlen; i++)
	{
		labels[i] = src[i] >= threshold ? +1.0 : -1.0;
	}
	set_labels(labels);
	set_values(src);
}

BinaryLabels::BinaryLabels(std::shared_ptr<File > loader) : DenseLabels(loader)
{
}

bool BinaryLabels::is_valid() const
{
	if (!DenseLabels::is_valid())
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

void BinaryLabels::ensure_valid(const char* context)
{
	require(is_valid(), "Binary Labels must be -1 or +1!");
}

ELabelType BinaryLabels::get_label_type() const
{
	return LT_BINARY;
}

void BinaryLabels::scores_to_probabilities(float64_t a, float64_t b)
{
	SG_TRACE("entering BinaryLabels::scores_to_probabilities()");

	require(m_current_values.vector, "{}::scores_to_probabilities() requires "
	        "values vector!", get_name());

	if (a == 0 && b == 0)
	{
		Statistics::SigmoidParamters params =
		        Statistics::fit_sigmoid(m_current_values);
		a = params.a;
		b = params.b;
	}

	SG_DEBUG("using sigmoid: a={}, b={}", a, b)

	/* now the sigmoid is fitted, convert all values to probabilities */
	for (index_t i = 0; i < m_current_values.vlen; ++i)
	{
		float64_t fApB = m_current_values[i] * a + b;
		m_current_values[i] = fApB >= 0
		                          ? std::exp(-fApB) / (1.0 + std::exp(-fApB))
		                          : 1.0 / (1 + std::exp(fApB));
	}

	SG_TRACE("leaving BinaryLabels::scores_to_probabilities()");
}

std::shared_ptr<Labels> BinaryLabels::shallow_subset_copy()
{
	SGVector<float64_t> shallow_copy_vector(m_labels);
	auto shallow_copy_labels=std::make_shared<BinaryLabels>(m_labels.size());


	shallow_copy_labels->set_labels(shallow_copy_vector);
	if (m_subset_stack->has_subsets())
		shallow_copy_labels->add_subset(m_subset_stack->get_last_subset()->get_subset_idx());

	return shallow_copy_labels;
}

BinaryLabels::BinaryLabels(const DenseLabels& dense) : DenseLabels(dense)
{
	ensure_valid();
}

std::shared_ptr<Labels> BinaryLabels::duplicate() const
{
	return std::make_shared<BinaryLabels>(*this);
}

namespace shogun
{
	std::shared_ptr<BinaryLabels> binary_labels(std::shared_ptr<Labels> orig)
	{
		require(orig, "No labels provided.");
		try
		{
			switch (orig->get_label_type())
			{
			case LT_BINARY:
				return std::static_pointer_cast<BinaryLabels>(orig);
			default:
				not_implemented(SOURCE_LOCATION);
			}
		}
		catch (const ShogunException& e)
		{
			error(
			    "Cannot convert {} to binary labels: {}", orig->get_name(),
			    e.what());
		}

		return nullptr;
	}
} // namespace shogun
