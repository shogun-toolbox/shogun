/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu 
 * Copyright (C) 2013 Shell Hu 
 */

#include <shogun/structure/FactorType.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

CFactorType::CFactorType() : CSGObject()
{
	SG_UNSTABLE("CFactorType::CFactorType()", "\n");

	init();
}

CFactorType::CFactorType(
	const SGString<char> id, 
	const SGVector<int32_t> card,
	const SGVector<float64_t> w)
	: CSGObject()
{
	init();
	m_type_id = id;
	m_w = w;
	m_cards = card;
	init_card();

	ASSERT(m_cards.size() > 0);

	if (m_w.size() == 0) 
	{
		m_data_size = m_prod_card;
	} 
	else 
	{
		ASSERT(m_w.size() % m_prod_card == 0);
		m_data_size = m_w.size() / m_prod_card;
	}

	m_map_params.resize_vector(m_w.size());
	m_map_params.zero();
}

CFactorType::~CFactorType() 
{
}

void CFactorType::init()
{
	SG_ADD(&m_type_id, "type_id", "Factor type name", MS_NOT_AVAILABLE);
	SG_ADD(&m_cards, "cards", "Cardinalities", MS_NOT_AVAILABLE);
	SG_ADD(&m_cumprod_cards, "cumprod_cards", "Cumulative product of cardinalities", MS_NOT_AVAILABLE);
	SG_ADD(&m_prod_card, "prod_card", "Product of cardinalities", MS_NOT_AVAILABLE);
	SG_ADD(&m_w, "w", "Factor parameters", MS_NOT_AVAILABLE);
	SG_ADD(&m_data_size, "data_size", "Size of data vector", MS_NOT_AVAILABLE);
	SG_ADD(&m_map_params, "map_params", "Map indices in global model parameters", MS_NOT_AVAILABLE);

	m_data_size = 0;
	m_prod_card = 0;
}

const char* CFactorType::get_type_id() const 
{
	return m_type_id.string;
}

void CFactorType::set_type_id(char* tid_str, int32_t slen)
{
	m_type_id = SGString<char>(tid_str, slen);
}

SGVector<float64_t> CFactorType::get_w() 
{
	return m_w;
}

const SGVector<float64_t> CFactorType::get_w() const 
{
	return m_w;
}

void CFactorType::set_w(SGVector<float64_t> w)
{
	m_w = w.clone();
}

int32_t CFactorType::get_w_dim() const 
{
	return m_w.size();
}

const SGVector<int32_t> CFactorType::get_cardinalities() const 
{
	return m_cards;
}

void CFactorType::set_cardinalities(SGVector<int32_t> cards)
{
	m_cards = cards.clone();
	init_card();
	if (m_w.size() == 0) 
	{
		m_data_size = m_prod_card;
	} 
	else 
	{
		ASSERT(m_w.size() % m_prod_card == 0);
		m_data_size = m_w.size() / m_prod_card;
	}
}

int32_t CFactorType::get_prodcardinalities() const 
{
	return m_prod_card;
}

void CFactorType::set_params_mapping(SGVector<int32_t> map)
{
	m_map_params = map.clone();	
}

void CFactorType::init_card() 
{
	m_prod_card = 1;
	m_cumprod_cards.resize_vector(m_cards.size());
	for (int32_t n = 0; n < m_cards.size(); ++n) 
	{
		m_cumprod_cards[n] = m_prod_card;
		m_prod_card *= m_cards[n];
	}
}

CTableFactorType::CTableFactorType()
	: CFactorType()
{
}

CTableFactorType::CTableFactorType(
	const SGString<char> id, 
	const SGVector<int32_t> card,
	const SGVector<float64_t> w)
	: CFactorType(id, card, w)
{
}

CTableFactorType::~CTableFactorType() 
{
}

int32_t CTableFactorType::state_from_index(int32_t ei, int32_t var_index) const 
{
	ASSERT(var_index < get_cardinalities().size());
	return ((ei / m_cumprod_cards[var_index]) % m_cards[var_index]);
}

SGVector<int32_t> CTableFactorType::assignment_from_index(int32_t ei) const
{
	SGVector<int32_t> assig(get_cardinalities().size());
	for (int32_t vi = 0; vi < get_cardinalities().size(); ++vi) 
		assig[vi] = state_from_index(ei, vi);

	return assig;
}

int32_t CTableFactorType::index_from_assignment(const SGVector<int32_t> assig) const
{
	ASSERT(assig.size() == get_cardinalities().size());
	int32_t index = 0;
	for (int32_t vi = 0; vi < get_cardinalities().size(); ++vi) 
		index += assig[vi] * m_cumprod_cards[vi];

	return index;
}

int32_t CTableFactorType::index_from_new_state(
	int32_t old_ei, int32_t var_index, int32_t var_state) const
{
	ASSERT(var_index < get_cardinalities().size());
	ASSERT(var_state < get_cardinalities()[var_index]);
	// subtract old contribution and add new contribution from new state
	return (old_ei - state_from_index(old_ei, var_index) * m_cumprod_cards[var_index] 
		+ var_state * m_cumprod_cards[var_index]);
}

int32_t CTableFactorType::index_from_universe_assignment(
	const SGVector<int32_t> assig,
	const SGVector<int32_t> var_index) const
{
	ASSERT(var_index.size() == m_cards.size());
	int32_t index = 0;
	for (int32_t vi = 0; vi < var_index.size(); vi++)
	{
		int32_t cur_var = var_index[vi];
		ASSERT(assig[cur_var] <= m_cards[vi]);
		index += assig[cur_var] * m_cumprod_cards[vi];  
	}
	ASSERT(index < m_prod_card);
	return index;
}

void CTableFactorType::compute_energies(
	const SGVector<float64_t> factor_data,
	SGVector<float64_t>& energies) const 
{
	ASSERT(energies.size() == m_prod_card);

	if (factor_data.size() == 0) 
	{
		ASSERT(m_w.size() == m_prod_card);
		energies = m_w.clone();
	} 
	else if (m_w.size() == 0) 
	{
		ASSERT(m_data_size == m_prod_card);
		ASSERT(factor_data.size() == m_data_size); 
		energies = factor_data.clone();
	} 
	else 
	{
		ASSERT(m_data_size * m_prod_card == m_w.size());
		ASSERT(m_data_size == factor_data.size());
		for (int32_t ei = 0; ei < m_prod_card; ++ei) 
		{
			float64_t energy_cur = 0.0;
			for (int32_t di = 0; di < m_data_size; ++di)
				energy_cur += factor_data[di] * m_w[di + ei*m_data_size];

			energies[ei] = energy_cur;
		}
	}
}

void CTableFactorType::compute_energies(
	const SGSparseVector<float64_t> factor_data_sparse,
	SGVector<float64_t>& energies) const 
{
	ASSERT(energies.size() == m_prod_card);
	ASSERT(m_data_size >= factor_data_sparse.num_feat_entries);

	if (factor_data_sparse.num_feat_entries == 0) 
	{
		ASSERT(m_prod_card == m_w.size());
		energies = m_w.clone();
	} 
	else if (m_w.size() == 0) 
	{
		ASSERT(m_prod_card == m_data_size);
		energies.zero();
		SGSparseVectorEntry<float64_t>* data_ptr = factor_data_sparse.features;
		for (int32_t n = 0; n < factor_data_sparse.num_feat_entries; ++n)
			energies[data_ptr[n].feat_index] = data_ptr[n].entry;
	} 
	else 
	{
		ASSERT((m_data_size * m_prod_card) == m_w.size());
		SGSparseVectorEntry<float64_t>* data_ptr = factor_data_sparse.features;

		for (int32_t ei = 0; ei < m_prod_card; ++ei) 
		{
			float64_t energy_cur = 0.0;
			for (int32_t n = 0; n < factor_data_sparse.num_feat_entries; ++n) 
			{
				energy_cur += data_ptr[n].entry 
					* m_w[data_ptr[n].feat_index + ei*m_data_size];
			}
			energies[ei] = energy_cur;
		}
	}
}

void CTableFactorType::compute_gradients(
	const SGVector<float64_t> factor_data,
	const SGVector<float64_t> marginals,
	SGVector<float64_t>& parameter_gradient, 
	double mult) const 
{
	if (factor_data.size() == 0) 
	{
		ASSERT(m_prod_card == parameter_gradient.size());
		// Parameters are a simple table, gradient is simply the marginal
		for (int32_t ei = 0; ei < m_prod_card; ++ei)
			parameter_gradient[ei] = mult * marginals[ei];
	} 
	else if (m_w.size() == 0) 
	{
		SG_ERROR("%s::compute_gradients(): no parameters for this factor type.\n", get_name());
	} 
	else 
	{
		ASSERT((m_data_size * m_prod_card) == parameter_gradient.size());
		ASSERT(m_data_size == factor_data.size());
		// Perform tensor outer product
		for (int32_t ei = 0; ei < m_prod_card; ++ei) 
		{
			for (int32_t di = 0; di < m_data_size; ++di) 
			{
				parameter_gradient[di + ei*m_data_size] +=
					mult * factor_data[di] * marginals[ei];
			}
		}
	}
}

void CTableFactorType::compute_gradients(
	const SGSparseVector<float64_t> factor_data_sparse,
	const SGVector<float64_t> marginals,
	SGVector<float64_t>& parameter_gradient, 
	double mult) const 
{
	// The first two cases are the same as for the non-sparse case
	if (factor_data_sparse.num_feat_entries == 0) 
	{
		ASSERT(m_prod_card == parameter_gradient.size());
		// Parameters are a simple table, gradient is simply the marginal
		for (int32_t ei = 0; ei < m_prod_card; ++ei)
			parameter_gradient[ei] = mult * marginals[ei];
	} 
	else if (m_w.size() == 0) 
	{
		SG_ERROR("%s::compute_gradients(): no parameters for this factor type.\n", get_name());
	} 
	else 
	{
		ASSERT((m_data_size * m_prod_card) == parameter_gradient.size());
		SGSparseVectorEntry<float64_t>* data_ptr = factor_data_sparse.features;

		// Perform tensor outer product
		for (int32_t ei = 0; ei < m_prod_card; ++ei) {
			for (int32_t n = 0; n < factor_data_sparse.num_feat_entries; ++n) {
				int32_t di = data_ptr[n].feat_index;
				parameter_gradient[di + ei*m_data_size] +=
					mult * data_ptr[n].entry * marginals[ei];
			}
		}
	}
}
