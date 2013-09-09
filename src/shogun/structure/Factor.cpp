/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu 
 * Copyright (C) 2013 Shell Hu 
 */

#include <shogun/structure/Factor.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

CFactor::CFactor() : CSGObject()
{
	SG_UNSTABLE("CFactor::CFactor()", "\n");
	init();
}

CFactor::CFactor(CTableFactorType* ftype,
	SGVector<int32_t> var_index,
	SGVector<float64_t> data) : CSGObject()
{
	init();
	m_factor_type = ftype;
	m_var_index = var_index;
	m_data = data;
	m_is_data_dep = true;

	ASSERT(ftype != NULL);
	ASSERT(ftype->get_cardinalities().size() == m_var_index.size());

	if (m_data.size() == 0)
		m_is_data_dep = false;

	if (ftype->is_table() && m_is_data_dep)
		m_energies.resize_vector(ftype->get_num_assignments());

	SG_REF(m_factor_type);
	SG_REF(m_data_source);
}

CFactor::CFactor(CTableFactorType* ftype,
	SGVector<int32_t> var_index,
	SGSparseVector<float64_t> data_sparse) : CSGObject()
{
	init();
	m_factor_type = ftype;
	m_var_index = var_index;
	m_data_sparse = data_sparse;
	m_is_data_dep = true;

	ASSERT(ftype != NULL);
	ASSERT(ftype->get_cardinalities().size() == m_var_index.size());

	if (m_data_sparse.num_feat_entries == 0)
		m_is_data_dep = false;

	if (ftype->is_table() && m_is_data_dep)
		m_energies.resize_vector(ftype->get_num_assignments());

	SG_REF(m_factor_type);
	SG_REF(m_data_source);
}

CFactor::CFactor(CTableFactorType* ftype,
	SGVector<int32_t> var_index,
	CFactorDataSource* data_source) : CSGObject()
{
	init();
	m_factor_type = ftype;
	m_var_index = var_index;
	m_data_source = data_source;
	m_is_data_dep = true;

	ASSERT(ftype != NULL);
	ASSERT(ftype->get_cardinalities().size() == m_var_index.size());
	ASSERT(data_source != NULL);

	if (ftype->is_table())
		m_energies.resize_vector(ftype->get_num_assignments());

	SG_REF(m_factor_type);
	SG_REF(m_data_source);
}

CFactor::~CFactor()
{
	SG_UNREF(m_factor_type);
	SG_UNREF(m_data_source);
}

CTableFactorType* CFactor::get_factor_type() const 
{
	SG_REF(m_factor_type);
	return m_factor_type;
}

void CFactor::set_factor_type(CTableFactorType* ftype)
{
	m_factor_type = ftype;
	SG_REF(m_factor_type);
}

const SGVector<int32_t> CFactor::get_variables() const 
{
	return m_var_index;
}

void CFactor::set_variables(SGVector<int32_t> vars)
{
	m_var_index = vars.clone();
}

const SGVector<int32_t> CFactor::get_cardinalities() const 
{
	return m_factor_type->get_cardinalities();
}

SGVector<float64_t> CFactor::get_data() const 
{
	if (m_data_source != NULL)
		return m_data_source->get_data();
	
	return m_data;
}

SGSparseVector<float64_t> CFactor::get_data_sparse() const 
{
	if (m_data_source != NULL)
		return m_data_source->get_data_sparse();

	return m_data_sparse;
}

void CFactor::set_data(SGVector<float64_t> data_dense)
{
	m_data = data_dense.clone();	
	m_is_data_dep = true;
}

void CFactor::set_data_sparse(SGSparseVectorEntry<float64_t>* data_sparse, 
	int32_t dlen)
{
	m_data_sparse = SGSparseVector<float64_t>(data_sparse, dlen);
	m_is_data_dep = true;
}

bool CFactor::is_data_dependent() const
{
	return m_is_data_dep;
}

bool CFactor::is_data_sparse() const
{
	if (m_data_source != NULL)
		return m_data_source->is_sparse();

	return (m_data.size() == 0);
}

SGVector<float64_t> CFactor::get_energies() const 
{
	if (is_data_dependent() == false && m_energies.size() == 0) 
	{
		const SGVector<float64_t> ft_energies = m_factor_type->get_w();
		ASSERT(ft_energies.size() == m_factor_type->get_num_assignments());
		return ft_energies;
	}
	return m_energies;
}

SGVector<float64_t> CFactor::get_energies() 
{
	if (is_data_dependent() == false && m_energies.size() == 0) 
	{
		SGVector<float64_t> ft_energies 
			= const_cast<CTableFactorType*>(m_factor_type)->get_w();
		ASSERT(ft_energies.size() == m_factor_type->get_num_assignments());
		return ft_energies;
	}
	return m_energies;
}

float64_t CFactor::get_energy(int32_t index) const
{
	return get_energies()[index]; // note for data indep, we get m_w not m_energies
}

void CFactor::set_energies(SGVector<float64_t> ft_energies)
{
	REQUIRE(m_factor_type->get_num_assignments() == ft_energies.size(),
		"%s::set_energies(): ft_energies is not a valid energy table!\n", get_name());

	m_energies = ft_energies;
}

void CFactor::set_energy(int32_t ei, float64_t value)
{
	REQUIRE(ei >= 0 && ei < m_factor_type->get_num_assignments(), 
		"%s::set_energy(): ei is out of index!\n", get_name());

	m_energies[ei] = value;
}

float64_t CFactor::evaluate_energy(const SGVector<int32_t> state) const 
{
	int32_t index = m_factor_type->index_from_universe_assignment(state, m_var_index);
	return get_energy(index); 
}

void CFactor::compute_energies() 
{
	if (is_data_dependent() == false) 
		return;

	// For some factor types the size of the energy table is determined only
	// after an initialization step from training data.
	if (m_energies.size() == 0)
		m_energies.resize_vector(m_factor_type->get_num_assignments());

	const SGVector<float64_t> H = get_data();
	const SGSparseVector<float64_t> H_sparse = get_data_sparse();

	if (H_sparse.num_feat_entries == 0) 
		m_factor_type->compute_energies(H, m_energies);
	else 
		m_factor_type->compute_energies(H_sparse, m_energies);
}

void CFactor::compute_gradients(
	const SGVector<float64_t> marginals,
	SGVector<float64_t>& parameter_gradient, 
	float64_t mult) const 
{
	const SGVector<float64_t> H = get_data();
	const SGSparseVector<float64_t> H_sparse = get_data_sparse();

	if (H_sparse.num_feat_entries == 0) 
		m_factor_type->compute_gradients(H, marginals, parameter_gradient, mult);
	else 
		m_factor_type->compute_gradients(H_sparse, marginals, parameter_gradient, mult);
}

void CFactor::init()
{
	SG_ADD((CSGObject**)&m_factor_type, "type_name", "Factor type name", MS_NOT_AVAILABLE);
	SG_ADD(&m_var_index, "var_index", "Factor variable index", MS_NOT_AVAILABLE);
	SG_ADD(&m_energies, "energies", "Factor energies", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_data_source, "data_source", "Factor data source", MS_NOT_AVAILABLE);
	SG_ADD(&m_data, "data", "Factor data", MS_NOT_AVAILABLE);
	SG_ADD(&m_data_sparse, "data_sparse", "Sparse factor data", MS_NOT_AVAILABLE);
	SG_ADD(&m_is_data_dep, "is_data_dep", "Factor is data dependent or not", MS_NOT_AVAILABLE);

	m_factor_type=NULL;
	m_data_source=NULL;
	m_is_data_dep = false;
}

CFactorDataSource::CFactorDataSource() : CSGObject()
{
	init();
}

CFactorDataSource::CFactorDataSource(SGVector<float64_t> dense)
	: CSGObject()
{
	init();
	m_dense = dense;
}

CFactorDataSource::CFactorDataSource(SGSparseVector<float64_t> sparse)
	: CSGObject()
{
	init();
	m_sparse = sparse;
}

CFactorDataSource::~CFactorDataSource() 
{
}

bool CFactorDataSource::is_sparse() const 
{
	return (m_dense.size() == 0);
}

SGVector<float64_t> CFactorDataSource::get_data() const 
{
	return m_dense;
}

SGSparseVector<float64_t> CFactorDataSource::get_data_sparse() const 
{
	return m_sparse;
}

void CFactorDataSource::set_data(SGVector<float64_t> dense)
{
	m_dense = dense.clone();	
}

void CFactorDataSource::set_data_sparse(SGSparseVectorEntry<float64_t>* sparse, 
	int32_t dlen)
{
	m_sparse = SGSparseVector<float64_t>(sparse, dlen);
}

void CFactorDataSource::init()
{
	SG_ADD(&m_dense, "dense", "Shared data", MS_NOT_AVAILABLE);
	SG_ADD(&m_sparse, "sparse", "Shared sparse data", MS_NOT_AVAILABLE);
}

