#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/RegressionLabels.h>

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

namespace shogun
{
	Some<CRegressionLabels> regression_from_regression(CRegressionLabels* orig)
	{
		return Some<CRegressionLabels>::from_raw(orig);
	}

	Some<CRegressionLabels> regression_from_dense(CDenseLabels* orig)
	{
		auto result = new CRegressionLabels(orig->get_labels());
		result->set_values(orig->get_values());
		return Some<CRegressionLabels>::from_raw(result);
	}

	Some<CRegressionLabels> regression_labels(CLabels* orig)
	{
		REQUIRE(orig, "No labels provided.\n");
		try
		{
			switch (orig->get_label_type())
			{
			case LT_REGRESSION:
				return regression_from_regression(
				    orig->as<CRegressionLabels>());
			case LT_DENSE_GENERIC:
				return regression_from_dense(orig->as<CDenseLabels>());
			default:
				SG_SNOTIMPLEMENTED
			}
		}
		catch (const ShogunException& e)
		{
			SG_SERROR(
			    "Cannot convert %s to regression labels: \n", e.what(),
			    orig->get_name());
		}

		return Some<CRegressionLabels>::from_raw(nullptr);
	}
} // namespace shogun
