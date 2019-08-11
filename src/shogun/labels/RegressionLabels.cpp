#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;

CRegressionLabels::CRegressionLabels() : CDenseLabels()
{
}

CRegressionLabels::CRegressionLabels(int32_t num_labels) : CDenseLabels(num_labels)
{
}

CRegressionLabels::CRegressionLabels(const SGVector<float64_t> src) : CDenseLabels()
{
	set_labels(src);
}

CRegressionLabels::CRegressionLabels(CFile* loader) : CDenseLabels(loader)
{
}

ELabelType CRegressionLabels::get_label_type() const
{
	return LT_REGRESSION;
}

CLabels* CRegressionLabels::shallow_subset_copy()
{
	CLabels* shallow_copy_labels=NULL;
	SGVector<float64_t> shallow_copy_vector(m_labels);
	shallow_copy_labels=new CRegressionLabels(m_labels.size());
	SG_REF(shallow_copy_labels);

	((CDenseLabels*) shallow_copy_labels)->set_labels(shallow_copy_vector);
	if (m_subset_stack->has_subsets())
		shallow_copy_labels->add_subset(m_subset_stack->get_last_subset()->get_subset_idx());

	return shallow_copy_labels;
}

CLabels* CRegressionLabels::duplicate() const
{
	return new CRegressionLabels(*this);
}

namespace shogun
{
	Some<CRegressionLabels> regression_labels(CLabels* orig)
	{
		require(orig, "No labels provided.\n");
		try
		{
			switch (orig->get_label_type())
			{
			case LT_REGRESSION:
				return Some<CRegressionLabels>::from_raw(
				    orig->as<CRegressionLabels>());
			case LT_BINARY:
				return some<CRegressionLabels>(
					orig->as<CBinaryLabels>()->get_labels());
			default:
				not_implemented(SOURCE_LOCATION);
			}
		}
		catch (const ShogunException& e)
		{
			error(
			    "Cannot convert {} to regression labels: \n", e.what(),
			    orig->get_name());
		}

		return Some<CRegressionLabels>::from_raw(nullptr);
	}
} // namespace shogun
