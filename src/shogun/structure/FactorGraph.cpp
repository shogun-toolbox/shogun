/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu 
 * Copyright (C) 2013 Shell Hu 
 */

#include <shogun/structure/FactorGraph.h>

using namespace shogun;

CFactorGraph::CFactorGraph() 
{
	SG_UNSTABLE("CFactorGraph::CFactorGraph()", "\n");

	register_parameters();
}

CFactorGraph::CFactorGraph(SGVector<int32_t> card)
	: m_cardinalities(card)
{
	register_parameters();
}

CFactorGraph::CFactorGraph(const CFactorGraph &fg)
{
	register_parameters();
	m_cardinalities = fg.get_cardinalities();
	// No need to unref and ref in this case
	// TODO test if need to copy element by element
	m_factors = fg.get_factors();
	m_datasources = fg.get_factor_data_sources();
}

CFactorGraph::~CFactorGraph() 
{
	SG_UNREF(m_factors);
	SG_UNREF(m_datasources);

	if (m_factors != NULL)
		SG_DEBUG("CFactorGraph::~CFactorGraph(): m_factors->ref_count() = %d.\n", m_factors->ref_count());

	if (m_datasources != NULL)
		SG_DEBUG("CFactorGraph::~CFactorGraph(): m_datasources->ref_count() = %d.\n", m_datasources->ref_count());

	SG_DEBUG("CFactorGraph::~CFactorGraph(): this->ref_count() = %d.\n", this->ref_count());
}

void CFactorGraph::register_parameters()
{
	SG_ADD(&m_cardinalities, "m_cardinalities", "Cardinalities", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_factors, "m_factors", "Factors", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_datasources, "m_datasources", "Factor data sources", MS_NOT_AVAILABLE);

	m_factors = NULL;
	m_datasources = NULL;
	m_factors = new CDynamicObjectArray();
	m_datasources = new CDynamicObjectArray();

	if (m_factors != NULL)
		SG_DEBUG("CFactorGraph::register_parameters(): m_factors->ref_count() = %d.\n", m_factors->ref_count());

	SG_REF(m_factors);
	SG_REF(m_datasources);
}

CDynamicObjectArray* CFactorGraph::get_factors() const
{
	SG_REF(m_factors);
	return m_factors;
}

SGVector<int32_t> CFactorGraph::get_cardinalities() const
{
	return m_cardinalities;
}

void CFactorGraph::set_cardinalities(SGVector<int32_t> cards)
{
	m_cardinalities = cards.clone();
}

CDynamicObjectArray* CFactorGraph::get_factor_data_sources() const
{
	SG_REF(m_datasources);
	return m_datasources;
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

void CFactorGraph::add_factor(CFactor* factor) 
{
	m_factors->push_back(factor);
}

void CFactorGraph::add_data_source(CFactorDataSource* datasource) 
{
	m_datasources->push_back(datasource);
}

CFactorGraph* CFactorGraph::duplicate() const
{
	return new CFactorGraph(*this);
}

int32_t CFactorGraph::get_num_vectors() const
{
	return m_factors->get_num_elements();
}

