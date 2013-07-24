#include <shogun/structure/FactorGraphLabels.h>

using namespace shogun;

CFactorGraphLabels::CFactorGraphLabels()
: CStructuredLabels()
{
}

CFactorGraphLabels::CFactorGraphLabels(int32_t num_labels, int32_t num_states)
: CStructuredLabels(num_labels), m_num_states(num_states)
{
	init();
}

CFactorGraphLabels::~CFactorGraphLabels()
{
}

void CFactorGraphLabels::init()
{
	SG_ADD(&m_num_states, "m_num_states", "Number of states", MS_NOT_AVAILABLE);
}
