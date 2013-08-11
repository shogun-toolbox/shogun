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

