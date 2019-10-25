#include <shogun/labels/FactorGraphLabels.h>

using namespace shogun;

FactorGraphObservation::FactorGraphObservation(SGVector<int32_t> observed_state,
	SGVector<float64_t> loss_weights)
	: StructuredData(), m_observed_state(observed_state)
{
	if (loss_weights.size() == 0)
	{
		loss_weights.resize_vector(observed_state.size());
		SGVector<float64_t>::fill_vector(loss_weights.vector, loss_weights.vlen, 1.0 / observed_state.size());
	}

	set_loss_weights(loss_weights);
}

SGVector<int32_t> FactorGraphObservation::get_data() const
{
	return m_observed_state;
}

SGVector<float64_t> FactorGraphObservation::get_loss_weights() const
{
	return m_loss_weights;
}

void FactorGraphObservation::set_loss_weights(SGVector<float64_t> loss_weights)
{
	require(loss_weights.size() == m_observed_state.size(), "{}::set_loss_weights(): \
		loss_weights should be the same length as observed_states", get_name());

	m_loss_weights = loss_weights;
}

//-------------------------------------------------------------------

FactorGraphLabels::FactorGraphLabels()
: StructuredLabels()
{
}

FactorGraphLabels::FactorGraphLabels(int32_t num_labels)
: StructuredLabels(num_labels)
{
	init();
}

FactorGraphLabels::~FactorGraphLabels()
{
}

void FactorGraphLabels::init()
{
}
