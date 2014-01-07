/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <structure/StateModel.h>

using namespace shogun;

CStateModel::CStateModel() : CSGObject()
{
	init();
}

CStateModel::~CStateModel()
{
}

int32_t CStateModel::get_num_states() const
{
	return m_num_states;
}

int32_t CStateModel::get_num_transmission_params() const
{
	return m_num_transmission_params;
}

void CStateModel::init()
{
	SG_ADD(&m_num_states, "m_num_states", "The number of states", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_transmission_params, "m_num_tranmission_params",
			"The number of tranmission parameters", MS_NOT_AVAILABLE);
	SG_ADD(&m_state_loss_mat, "m_state_loss_mat", "The state loss matrix",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_p, "m_p", "The distribution of start states", MS_NOT_AVAILABLE);
	SG_ADD(&m_q, "m_q", "The distribution of stop states", MS_NOT_AVAILABLE);

	m_num_states = 0;
	m_num_transmission_params = 0;
}

SGVector< int32_t > CStateModel::get_monotonicity(int32_t num_free_states,
		int32_t num_feats) const
{
	SGVector< int32_t > ret(num_feats*num_free_states);
	ret.zero();
	return ret;
}

SGVector< float64_t > CStateModel::get_start_states() const
{
	return m_p;
}

SGVector< float64_t > CStateModel::get_stop_states() const
{
	return m_q;
}
