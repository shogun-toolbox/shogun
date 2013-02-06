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

CRegressionLabels* CRegressionLabels::obtain_from_generic(CLabels* base_labels)
{
	if ( base_labels->get_label_type() == LT_REGRESSION )
		return (CRegressionLabels*) base_labels;
	else
		SG_SERROR("base_labels must be of dynamic type CRegressionLabels")

	return NULL;
}

ELabelType CRegressionLabels::get_label_type()
{
	return LT_REGRESSION;
}

