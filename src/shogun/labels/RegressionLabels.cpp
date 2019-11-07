#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

#include <utility>

using namespace shogun;

RegressionLabels::RegressionLabels() : DenseLabels()
{
}

RegressionLabels::RegressionLabels(int32_t num_labels) : DenseLabels(num_labels)
{
}

RegressionLabels::RegressionLabels(SGVector<float64_t> src) : DenseLabels()
{
	set_labels(src);
}

RegressionLabels::RegressionLabels(std::shared_ptr<File> loader) : DenseLabels(std::move(loader))
{
}

ELabelType RegressionLabels::get_label_type() const
{
	return LT_REGRESSION;
}

std::shared_ptr<Labels> RegressionLabels::shallow_subset_copy()
{
	SGVector<float64_t> shallow_copy_vector(m_labels);
	auto shallow_copy_labels=std::make_shared<RegressionLabels>(m_labels.size());

	shallow_copy_labels->set_labels(shallow_copy_vector);
	if (m_subset_stack->has_subsets())
		shallow_copy_labels->add_subset(m_subset_stack->get_last_subset()->get_subset_idx());

	return shallow_copy_labels;
}

std::shared_ptr<Labels> RegressionLabels::duplicate() const
{
	return std::make_shared<RegressionLabels>(*this);
}

namespace shogun
{
	std::shared_ptr<RegressionLabels> regression_labels(const std::shared_ptr<Labels>& orig)
	{
		require(orig, "No labels provided.");
		try
		{
			switch (orig->get_label_type())
			{
			case LT_REGRESSION:
				return std::static_pointer_cast<RegressionLabels>(orig);
			case LT_BINARY:
				return std::make_shared<RegressionLabels>(
					(std::static_pointer_cast<BinaryLabels>(orig))->get_labels());
			default:
				not_implemented(SOURCE_LOCATION);
			}
		}
		catch (const ShogunException& e)
		{
			error(
			    "Cannot convert {} to regression labels: ", e.what(),
			    orig->get_name());
		}

		return nullptr;
	}
} // namespace shogun
