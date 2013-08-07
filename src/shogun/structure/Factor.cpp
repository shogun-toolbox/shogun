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

CFactor::CFactor()
	: m_factor_type(NULL), m_data_source(NULL) 
{
	SG_UNSTABLE("CFactor::CFactor()", "\n");
	register_parameters();
	m_is_data_dep = false;

	SG_REF(m_factor_type);
	SG_REF(m_data_source);
}

CFactor::CFactor(CTableFactorType* ftype,
	SGVector<int32_t> var_index,
	SGVector<float64_t> data)
	: m_factor_type(ftype), m_var_index(var_index), 
	m_data_source(NULL), m_data(data), m_is_data_dep(true)
{
	ASSERT(ftype != NULL);
	ASSERT(ftype->get_cardinalities().size() == m_var_index.size());

	register_parameters();

	if (m_data.size() == 0)
		m_is_data_dep = false;

	if (ftype->is_table() && m_is_data_dep)
		m_energies.resize_vector(ftype->get_prodcardinalities());

	SG_REF(m_factor_type);
	SG_REF(m_data_source);
}

CFactor::CFactor(CTableFactorType* ftype,
	SGVector<int32_t> var_index,
	SGSparseVector<float64_t> data_sparse)
	: m_factor_type(ftype), m_var_index(var_index), 
	m_data_source(NULL), m_data_sparse(data_sparse), m_is_data_dep(true) 
{
	ASSERT(ftype != NULL);
	ASSERT(ftype->get_cardinalities().size() == m_var_index.size());

	register_parameters();

	if (m_data_sparse.num_feat_entries == 0)
		m_is_data_dep = false;

	if (ftype->is_table() && m_is_data_dep)
		m_energies.resize_vector(ftype->get_prodcardinalities());

	SG_REF(m_factor_type);
	SG_REF(m_data_source);
}

CFactor::CFactor(CTableFactorType* ftype,
	SGVector<int32_t> var_index,
	CFactorDataSource* data_source)
	: m_factor_type(ftype), m_var_index(var_index), 
	m_data_source(data_source), m_is_data_dep(true) 
{
	ASSERT(ftype != NULL);
	ASSERT(ftype->get_cardinalities().size() == m_var_index.size());
	ASSERT(data_source != NULL);

	register_parameters();

	if (ftype->is_table())
		m_energies.resize_vector(ftype->get_prodcardinalities());

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

const SGVector<float64_t> CFactor::get_data() const 
{
	if (m_data_source != NULL)
		return m_data_source->data();

	return m_data;
}

const SGSparseVector<float64_t> CFactor::get_data_sparse() const 
{
	if (m_data_source != NULL)
		return m_data_source->data_sparse();

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

const SGVector<float64_t> CFactor::get_energies() const 
{
	if (is_data_dependent() == false && m_energies.size() == 0) 
	{
		const SGVector<float64_t> ft_energies = m_factor_type->get_w();
		ASSERT(ft_energies.size() == m_factor_type->get_prodcardinalities());
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
		ASSERT(ft_energies.size() == m_factor_type->get_prodcardinalities());
		return ft_energies;
	}
	return m_energies;
}

float64_t CFactor::evaluate_energy(const SGVector<int32_t> state) const 
{
	int32_t index = m_factor_type->index_from_universe_assignment(state, m_var_index);
	return get_energies()[index]; // note for data indep, we get m_w not m_energies
}

void CFactor::compute_energies() 
{
	if (is_data_dependent() == false) 
		return;

	// For some factor types the size of the energy table is determined only
	// after an initialization step from training data.
	if (m_energies.size() == 0)
		m_energies.resize_vector(m_factor_type->get_prodcardinalities());

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

void CFactor::register_parameters()
{
	SG_ADD((CSGObject**)&m_factor_type, "m_type_name", "Factor type name", MS_NOT_AVAILABLE);
	SG_ADD(&m_var_index, "m_var_index", "Factor variable index", MS_NOT_AVAILABLE);
	SG_ADD(&m_energies, "m_energies", "Factor energies", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_data_source, "m_data_source", "Factor data source", MS_NOT_AVAILABLE);
	SG_ADD(&m_data, "m_data", "Factor data", MS_NOT_AVAILABLE);
	SG_ADD(&m_data_sparse, "m_data_sparse", "Sparse factor data", MS_NOT_AVAILABLE);
	SG_ADD(&m_is_data_dep, "m_is_data_dep", "Factor is data dependent or not", MS_NOT_AVAILABLE);
	SG_ADD(&m_factor_id, "m_factor_id", "Factor ID in the factor graph", MS_NOT_AVAILABLE);
}

CFactorDataSource::CFactorDataSource() 
{
	register_parameters();
}

CFactorDataSource::CFactorDataSource(const SGVector<float64_t> dense)
	: m_dense(dense) 
{
	register_parameters();
}

CFactorDataSource::CFactorDataSource(const SGSparseVector<float64_t> sparse)
	: m_sparse(sparse)
{
	register_parameters();
}

CFactorDataSource::~CFactorDataSource() 
{
}

bool CFactorDataSource::is_sparse() const 
{
	return (m_dense.size() == 0);
}

const SGVector<float64_t> CFactorDataSource::data() const 
{
	return m_dense;
}

const SGSparseVector<float64_t> CFactorDataSource::data_sparse() const 
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

void CFactorDataSource::register_parameters()
{
	SG_ADD(&m_dense, "m_dense", "Shared data", MS_NOT_AVAILABLE);
	SG_ADD(&m_sparse, "m_sparse", "Shared sparse data", MS_NOT_AVAILABLE);
}

