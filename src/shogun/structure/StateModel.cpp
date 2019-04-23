/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias
 */

#include <shogun/structure/StateModel.h>

using namespace shogun;

StateModel::StateModel() : SGObject()
{
	init();
}

StateModel::~StateModel()
{
}

int32_t StateModel::get_num_states() const
{
	return m_num_states;
}

int32_t StateModel::get_num_transmission_params() const
{
	return m_num_transmission_params;
}

void StateModel::init()
{
	SG_ADD(&m_num_states, "m_num_states", "The number of states");
	SG_ADD(&m_num_transmission_params, "m_num_tranmission_params",
			"The number of tranmission parameters");
	SG_ADD(&m_state_loss_mat, "m_state_loss_mat", "The state loss matrix");
	SG_ADD(&m_p, "m_p", "The distribution of start states");
	SG_ADD(&m_q, "m_q", "The distribution of stop states");

	m_num_states = 0;
	m_num_transmission_params = 0;
}

SGVector< int32_t > StateModel::get_monotonicity(int32_t num_free_states,
		int32_t num_feats) const
{
	SGVector< int32_t > ret(num_feats*num_free_states);
	ret.zero();
	return ret;
}

SGVector< float64_t > StateModel::get_start_states() const
{
	return m_p;
}

SGVector< float64_t > StateModel::get_stop_states() const
{
	return m_q;
}
