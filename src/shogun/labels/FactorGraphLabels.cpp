#include <shogun/labels/FactorGraphLabels.h>

using namespace shogun;

CFactorGraphLabels::CFactorGraphLabels()
: CStructuredLabels()
{
}

CFactorGraphLabels::CFactorGraphLabels(int32_t num_labels)
: CStructuredLabels(num_labels)
{
	init();
}

CFactorGraphLabels::~CFactorGraphLabels()
{
}

void CFactorGraphLabels::init()
{
}
