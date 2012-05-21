#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/RealLabels.h>

using namespace shogun;

CRealLabels::CRealLabels() : CDenseLabels()
{
}

CRealLabels::CRealLabels(int32_t num_labels) : CDenseLabels(num_labels)
{
}

CRealLabels::CRealLabels(const SGVector<float64_t> src) : CDenseLabels()
{
	set_labels(src);
}

CRealLabels::CRealLabels(CFile* loader) : CDenseLabels(loader)
{
}

CRealLabels* CRealLabels::obtain_from_generic(CLabels* base_labels)
{
	if ( base_labels->get_label_type() == LT_REAL )
		return (CRealLabels*) base_labels;
	else
		SG_SERROR("base_labels must be of dynamic type CRealLabels");

	return NULL;
}

bool CRealLabels::is_valid()
{       
	ASSERT(m_labels.vector);
	return true;
}

ELabelType CRealLabels::get_label_type()
{
	return LT_REAL;
}

