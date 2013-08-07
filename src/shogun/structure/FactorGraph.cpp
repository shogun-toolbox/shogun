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

CFactorGraph::CDisjointSet::CDisjointSet()
	: m_num_elements(-1), m_parent(), m_rank()
{
	SG_UNSTABLE("CDisjointSet::CDisjointSet()", "\n");

	register_parameters();
}

CFactorGraph::CDisjointSet::CDisjointSet(int32_t num_elements)
	: m_num_elements(num_elements), m_parent(num_elements), m_rank(num_elements),
	m_is_connected(false)
{
	register_parameters();
}

void CFactorGraph::CDisjointSet::register_parameters()
{
	SG_ADD(&m_num_elements, "m_num_elements", "Number of elements", MS_NOT_AVAILABLE);
	SG_ADD(&m_parent, "m_parent", "Parent pointers", MS_NOT_AVAILABLE);
	SG_ADD(&m_rank, "m_rank", "Rank of each element", MS_NOT_AVAILABLE);
	SG_ADD(&m_is_connected, "m_is_connected", "Whether disjoint sets have been linked", MS_NOT_AVAILABLE);
}

void CFactorGraph::CDisjointSet::make_sets()
{
	REQUIRE(m_num_elements > 0, "%s::make_sets(): m_num_elements <= 0.\n", get_name());

	m_parent.range_fill();
	m_rank.zero();	
}

int32_t CFactorGraph::CDisjointSet::find_set(int32_t x) 
{
	ASSERT(x >= 0 && x < m_num_elements);

	// path compression
	if (x != m_parent[x])
		m_parent[x] = find_set(m_parent[x]); 

	return m_parent[x];
}

int32_t CFactorGraph::CDisjointSet::link_set(int32_t xroot, int32_t yroot)
{
	ASSERT(xroot >= 0 && xroot < m_num_elements);
	ASSERT(yroot >= 0 && yroot < m_num_elements);
	ASSERT(m_parent[xroot] == xroot && m_parent[yroot] == yroot);
	ASSERT(xroot != yroot);

	// union by rank
	if (m_rank[xroot] > m_rank[yroot])
	{
		m_parent[yroot] = xroot;
		return xroot;
	}
	else
	{
		m_parent[xroot] = yroot;
		if (m_rank[xroot] == m_rank[yroot])
			m_rank[yroot] += 1;

		return yroot;
	}
}

bool CFactorGraph::CDisjointSet::union_set(int32_t x, int32_t y) 
{
	ASSERT(x >= 0 && x < m_num_elements);
	ASSERT(y >= 0 && y < m_num_elements);

	int32_t xroot = find_set(x);
	int32_t yroot = find_set(y);

	if (xroot == yroot)
		return true;

	link_set(xroot, yroot);	
	return false;
}

bool CFactorGraph::CDisjointSet::is_same_set(int32_t x, int32_t y) 
{
	ASSERT(x >= 0 && x < m_num_elements);
	ASSERT(y >= 0 && y < m_num_elements);

	if (find_set(x) == find_set(y))
		return true;

	return false;
}

int32_t CFactorGraph::CDisjointSet::get_unique_labeling(SGVector<int32_t> out_labels) 
{
	REQUIRE(m_num_elements > 0, "%s::get_unique_labeling(): m_num_elements <= 0.\n", get_name());

	if (out_labels.size() != m_num_elements)
		out_labels.resize_vector(m_num_elements);

	SGVector<int32_t> roots(m_num_elements);
	SGVector<int32_t> flags(m_num_elements);
	SGVector<int32_t>::fill_vector(flags.vector, flags.vlen, -1);
	int32_t unilabel = 0;

	for (int32_t i = 0; i < m_num_elements; i++)
	{
		roots[i] = find_set(i);
		// if roots[i] never be found
		if (flags[roots[i]] < 0)
			flags[roots[i]] = unilabel++;

	}

	for (int32_t i = 0; i < m_num_elements; i++)
		out_labels[i] = flags[roots[i]];

	return unilabel;
}

int32_t CFactorGraph::CDisjointSet::get_num_sets() 
{
	REQUIRE(m_num_elements > 0, "%s::get_num_sets(): m_num_elements <= 0.\n", get_name());

	return get_unique_labeling(SGVector<int32_t>(m_num_elements));
}

bool CFactorGraph::CDisjointSet::get_connected() 
{
	return m_is_connected;
}

void CFactorGraph::CDisjointSet::set_connected(bool is_connected) 
{
	m_is_connected = is_connected;
}

CFactorGraph::CFactorGraph() 
{
	SG_UNSTABLE("CFactorGraph::CFactorGraph()", "\n");

	register_parameters();
	init();
}

CFactorGraph::CFactorGraph(SGVector<int32_t> card)
	: m_cardinalities(card)
{
	register_parameters();
	init();
}

CFactorGraph::CFactorGraph(const CFactorGraph &fg)
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
	SG_ADD(&m_cardinalities, "m_cardinalities", "Cardinalities", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_factors, "m_factors", "Factors", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_datasources, "m_datasources", "Factor data sources", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_dset, "m_dset", "Disjoint set", MS_NOT_AVAILABLE);
	SG_ADD(&m_has_cycle, "m_has_cycle", "Whether has circle in graph", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_edges, "m_num_edges", "Number of edges", MS_NOT_AVAILABLE);
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

SGVector<int32_t> CFactorGraph::get_cardinalities() const
{
	return m_cardinalities;
}

void CFactorGraph::set_cardinalities(SGVector<int32_t> cards)
{
	m_cardinalities = cards.clone();
}

CFactorGraph::CDisjointSet* CFactorGraph::get_disjoint_set() const
{
	SG_REF(m_dset);
	return m_dset;
}

int32_t CFactorGraph::get_num_edges() const
{
	return m_num_edges;
}

int32_t CFactorGraph::get_num_vectors() const
{
	return m_factors->get_num_elements();
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

CFactorGraph* CFactorGraph::duplicate() const
{
	return new CFactorGraph(*this);
}

// make sure call this func before other funcs related to m_dset
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

