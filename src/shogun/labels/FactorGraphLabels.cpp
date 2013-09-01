#include <shogun/labels/FactorGraphLabels.h>

using namespace shogun;

CFactorGraphObservation::CFactorGraphObservation(SGVector<int32_t> observed_state, 
	SGVector<float64_t> loss_weights)
	: CStructuredData(), m_observed_state(observed_state) 
{ 
	if (loss_weights.size() == 0)
	{
		loss_weights.resize_vector(observed_state.size());
		SGVector<float64_t>::fill_vector(loss_weights.vector, loss_weights.vlen, 1.0);	
	}

	REQUIRE(loss_weights.size() == observed_state.size(), "%s::CFactorGraphObservation(): \
		loss_weights should be the same length as observed_states", get_name());

	m_loss_weights = loss_weights;
}

SGVector<int32_t> CFactorGraphObservation::get_data() const 
{ 
	return m_observed_state; 
}

SGVector<float64_t> CFactorGraphObservation::get_loss_weights() const 
{ 
	return m_loss_weights; 
}

//-------------------------------------------------------------------

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
