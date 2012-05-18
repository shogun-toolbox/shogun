#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/RealLabels.h>

using namespace shogun;

CRealLabels::CRealLabels() : CDenseLabels()
{
}

CRealLabels::CRealLabels(int32_t num_labels) : CDenseLabels(num_labels)
{
}

CRealLabels::CRealLabels(const SGVector<float64_t> src) : CDenseLabels(src)
{
}

CRealLabels::CRealLabels(CFile* loader) : CDenseLabels(loader)
{
}

bool CRealLabels::is_valid()
{       
    ASSERT(labels.vector);
    return true;
}

ELabelType CRealLabels::get_label_type()
{
	return LT_REAL;
}

