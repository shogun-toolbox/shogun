/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu
 * Copyright (C) 2013 Shell Hu
 */

#include <structure/FactorGraph.h>
#include <labels/FactorGraphLabels.h>

using namespace shogun;

CFactorGraph::CFactorGraph()
	: CSGObject()
{
	SG_UNSTABLE("CFactorGraph::CFactorGraph()", "\n");

	register_parameters();
	init();
}

CFactorGraph::CFactorGraph(SGVector<int32_t> card)
	: CSGObject()
{
	m_cardinalities = card;
	register_parameters();
	init();
}

CFactorGraph::CFactorGraph(const CFactorGraph &fg)
	: CSGObject()
{
	register_parameters();
	m_cardinalities = fg.get_cardinalities();
	// No need to unref and ref in this case
	m_factors = fg.get_factors();
	m_datasources = fg.get_factor_data_sources();
	m_dset = fg.get_disjoint_set();
	m_has_cycle = !(fg.is_acyclic_graph());
	m_num_edges = fg.get_num_edges();
}

CFactorGraph::~CFactorGraph()
{
	SG_UNREF(m_factors);
	SG_UNREF(m_datasources);
	SG_UNREF(m_dset);

#ifdef USE_REFERENCE_COUNTING
	if (m_factors != NULL)
		SG_DEBUG("CFactorGraph::~CFactorGraph(): m_factors->ref_count() = %d.\n", m_factors->ref_count());

	if (m_datasources != NULL)
		SG_DEBUG("CFactorGraph::~CFactorGraph(): m_datasources->ref_count() = %d.\n", m_datasources->ref_count());

	SG_DEBUG("CFactorGraph::~CFactorGraph(): this->ref_count() = %d.\n", this->ref_count());
#endif
}

void CFactorGraph::register_parameters()
{
	SG_ADD(&m_cardinalities, "cardinalities", "Cardinalities", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_factors, "factors", "Factors", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_datasources, "datasources", "Factor data sources", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_dset, "dset", "Disjoint set", MS_NOT_AVAILABLE);
	SG_ADD(&m_has_cycle, "has_cycle", "Whether has circle in graph", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_edges, "num_edges", "Number of edges", MS_NOT_AVAILABLE);
}

void CFactorGraph::init()
{
	m_has_cycle = false;
	m_num_edges = 0;
	m_factors = NULL;
	m_datasources = NULL;
	m_factors = new CDynamicObjectArray();
	m_datasources = new CDynamicObjectArray();

#ifdef USE_REFERENCE_COUNTING
	if (m_factors != NULL)
		SG_DEBUG("CFactorGraph::init(): m_factors->ref_count() = %d.\n", m_factors->ref_count());
#endif

	// NOTE m_cards cannot be empty
	m_dset = new CDisjointSet(m_cardinalities.size());

	SG_REF(m_factors);
	SG_REF(m_datasources);
	SG_REF(m_dset);
}

void CFactorGraph::add_factor(CFactor* factor)
{
	m_factors->push_back(factor);
	m_num_edges += factor->get_variables().size();

	// graph structure changed after adding factors
	if (m_dset->get_connected())
		m_dset->set_connected(false);
}

void CFactorGraph::add_data_source(CFactorDataSource* datasource)
{
	m_datasources->push_back(datasource);
}

CDynamicObjectArray* CFactorGraph::get_factors() const
{
	SG_REF(m_factors);
	return m_factors;
}

CDynamicObjectArray* CFactorGraph::get_factor_data_sources() const
{
	SG_REF(m_datasources);
	return m_datasources;
}

int32_t CFactorGraph::get_num_factors() const
{
	return m_factors->get_num_elements();
}

SGVector<int32_t> CFactorGraph::get_cardinalities() const
{
	return m_cardinalities;
}

void CFactorGraph::set_cardinalities(SGVector<int32_t> cards)
{
	m_cardinalities = cards.clone();
}

CDisjointSet* CFactorGraph::get_disjoint_set() const
{
	SG_REF(m_dset);
	return m_dset;
}

int32_t CFactorGraph::get_num_edges() const
{
	return m_num_edges;
}

int32_t CFactorGraph::get_num_vars() const
{
	return m_cardinalities.size();
}

void CFactorGraph::compute_energies()
{
	for (int32_t fi = 0; fi < m_factors->get_num_elements(); ++fi)
	{
		CFactor* fac = dynamic_cast<CFactor*>(m_factors->get_element(fi));
		fac->compute_energies();
		SG_UNREF(fac);
	}
}

float64_t CFactorGraph::evaluate_energy(const SGVector<int32_t> state) const
{
	ASSERT(state.size() == m_cardinalities.size());

	float64_t energy = 0.0;
	for (int32_t fi = 0; fi < m_factors->get_num_elements(); ++fi)
	{
		CFactor* fac = dynamic_cast<CFactor*>(m_factors->get_element(fi));
		energy += fac->evaluate_energy(state);
		SG_UNREF(fac);
	}
	return energy;
}

float64_t CFactorGraph::evaluate_energy(const CFactorGraphObservation* obs) const
{
	return evaluate_energy(obs->get_data());
}

SGVector<float64_t> CFactorGraph::evaluate_energies() const
{
	int num_assig = 1;
	SGVector<int32_t> cumprod_cards(m_cardinalities.size());
	for (int32_t n = 0; n < m_cardinalities.size(); ++n)
	{
		cumprod_cards[n] = num_assig;
		num_assig *= m_cardinalities[n];
	}

	SGVector<float64_t> etable(num_assig);
	for (int32_t ei = 0; ei < num_assig; ++ei)
	{
		SGVector<int32_t> assig(m_cardinalities.size());
		for (int32_t vi = 0; vi < m_cardinalities.size(); ++vi)
			assig[vi] = (ei / cumprod_cards[vi]) % m_cardinalities[vi];

		etable[ei] = evaluate_energy(assig);

		for (int32_t vi = 0; vi < m_cardinalities.size(); ++vi)
			SG_SPRINT("%d ", assig[vi]);

		SG_SPRINT("| %f\n", etable[ei]);
	}

	return etable;
}

void CFactorGraph::connect_components()
{
	if (m_dset->get_connected())
		return;

	// need to be reset once factor graph is updated
	m_dset->make_sets();
	bool flag = false;

	for (int32_t fi = 0; fi < m_factors->get_num_elements(); ++fi)
	{
		CFactor* fac = dynamic_cast<CFactor*>(m_factors->get_element(fi));
		SGVector<int32_t> vars = fac->get_variables();

		int32_t r0 = m_dset->find_set(vars[0]);
		for (int32_t vi = 1; vi < vars.size(); vi++)
		{
			// for two nodes in a factor, should be an edge between them
			// but this time link() isn't performed, if they are linked already
			// means there is another path connected them, so cycle detected
			int32_t ri = m_dset->find_set(vars[vi]);

			if (r0 == ri)
			{
				flag = true;
				continue;
			}

			r0 = m_dset->link_set(r0, ri);
		}

		SG_UNREF(fac);
	}
	m_has_cycle = flag;
	m_dset->set_connected(true);
}

bool CFactorGraph::is_acyclic_graph() const
{
	return !m_has_cycle;
}

bool CFactorGraph::is_connected_graph() const
{
	return (m_dset->get_num_sets() == 1);
}

bool CFactorGraph::is_tree_graph() const
{
	return (m_has_cycle == false && m_dset->get_num_sets() == 1);
}

void CFactorGraph::loss_augmentation(CFactorGraphObservation* gt)
{
	loss_augmentation(gt->get_data(), gt->get_loss_weights());
}

void CFactorGraph::loss_augmentation(SGVector<int32_t> states_gt, SGVector<float64_t> loss)
{
	if (loss.size() == 0)
	{
		loss.resize_vector(states_gt.size());
		SGVector<float64_t>::fill_vector(loss.vector, loss.vlen, 1.0 / states_gt.size());
	}

	int32_t num_vars = states_gt.size();
	ASSERT(num_vars == loss.size());

	SGVector<int32_t> var_flags(num_vars);
	var_flags.zero();

	// augment loss to incorrect states in the first factor containing the variable
	// since += L_i for each variable if it takes wrong state ever
	// TODO: augment unary factors
	for (int32_t fi = 0; fi < m_factors->get_num_elements(); ++fi)
	{
		CFactor* fac = dynamic_cast<CFactor*>(m_factors->get_element(fi));
		SGVector<int32_t> vars = fac->get_variables();
		for (int32_t vi = 0; vi < vars.size(); vi++)
		{
			int32_t vv = vars[vi];
			ASSERT(vv < num_vars);
			if (var_flags[vv])
				continue;

			SGVector<float64_t> energies = fac->get_energies();
			for (int32_t ei = 0; ei < energies.size(); ei++)
			{
				CTableFactorType* ftype = fac->get_factor_type();
				int32_t vstate = ftype->state_from_index(ei, vi);
				SG_UNREF(ftype);

				if (states_gt[vv] == vstate)
					continue;

				// -delta(y_n, y_i_n)
				fac->set_energy(ei, energies[ei] - loss[vv]);
			}

			var_flags[vv] = 1;
		}

		SG_UNREF(fac);
	}

	// make sure all variables have been checked
	int32_t min_var = SGVector<int32_t>::min(var_flags.vector, var_flags.vlen);
	ASSERT(min_var == 1);
}

