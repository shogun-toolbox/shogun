/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu
 * Copyright (C) 2013 Shell Hu
 */

#include <shogun/structure/FactorGraphModel.h>
#include <shogun/structure/Factor.h>
#include <shogun/features/FactorGraphFeatures.h>

#ifdef HAVE_STD_UNORDERED_MAP
	#include <unordered_map>
	typedef std::unordered_map<int32_t, int32_t> factor_counts_type;
#else
	#include <tr1/unordered_map>
	typedef std::tr1::unordered_map<int32_t, int32_t> factor_counts_type;
#endif

using namespace shogun;

CFactorGraphModel::CFactorGraphModel()
	: CStructuredModel()
{
	init();
}

CFactorGraphModel::CFactorGraphModel(CFeatures* features, CStructuredLabels* labels,
	EMAPInferType inf_type, bool verbose) : CStructuredModel(features, labels)
{
	init();
	m_inf_type = inf_type;
	m_verbose = verbose;
}

CFactorGraphModel::~CFactorGraphModel()
{
	SG_UNREF(m_factor_types);
}

void CFactorGraphModel::init()
{
	SG_ADD((CSGObject**)&m_factor_types, "factor_types", "Array of factor types", MS_NOT_AVAILABLE);
	SG_ADD(&m_w_cache, "w_cache", "Cache of global parameters", MS_NOT_AVAILABLE);
	SG_ADD(&m_w_map, "w_map", "Parameter mapping", MS_NOT_AVAILABLE);

	m_inf_type = TREE_MAX_PROD;
	m_factor_types = new CDynamicObjectArray();
	m_verbose = false;

	SG_REF(m_factor_types);
}

void CFactorGraphModel::add_factor_type(CFactorType* ftype)
{
	REQUIRE(ftype->get_w_dim() > 0, "%s::add_factor_type(): number of parameters can't be 0!\n",
		get_name());

	// check whether this ftype has been added
	int32_t id = ftype->get_type_id();
	for (int32_t fi = 0; fi < m_factor_types->get_num_elements(); ++fi)
	{
		CFactorType* ft= dynamic_cast<CFactorType*>(m_factor_types->get_element(fi));
		if (id == ft->get_type_id())
		{
			SG_UNREF(ft);
			SG_PRINT("%s::add_factor_type(): factor_type (id = %d) has been added!\n",
				get_name(), id);

			return;
		}

		SG_UNREF(ft);
	}

	SGVector<int32_t> w_map_cp = m_w_map.clone();
	m_w_map.resize_vector(w_map_cp.size() + ftype->get_w_dim());

	for (int32_t mi = 0; mi < w_map_cp.size(); mi++)
	{
		m_w_map[mi] = w_map_cp[mi];
	}
	// add new mapping in the end
	for (int32_t mi = w_map_cp.size(); mi < m_w_map.size(); mi++)
	{
		m_w_map[mi] = id;
	}

	// push factor type
	m_factor_types->push_back(ftype);

	// initialize w cache
	fparams_to_w();

	if (m_verbose)
	{
		m_w_map.display_vector("add_factor_type(): m_w_map");
	}
}

void CFactorGraphModel::del_factor_type(const int32_t ftype_id)
{
	int w_dim = 0;
	// delete from m_factor_types
	for (int32_t fi = 0; fi < m_factor_types->get_num_elements(); ++fi)
	{
		CFactorType* ftype = dynamic_cast<CFactorType*>(m_factor_types->get_element(fi));
		if (ftype_id == ftype->get_type_id())
		{
			w_dim = ftype->get_w_dim();
			SG_UNREF(ftype);
			m_factor_types->delete_element(fi);
			break;
		}

		SG_UNREF(ftype);
	}

	ASSERT(w_dim != 0);

	SGVector<int32_t> w_map_cp = m_w_map.clone();
	m_w_map.resize_vector(w_map_cp.size() - w_dim);

	int ind = 0;
	for (int32_t mi = 0; mi < w_map_cp.size(); mi++)
	{
		if (w_map_cp[mi] == ftype_id)
			continue;

		m_w_map[ind++] = w_map_cp[mi];
	}

	ASSERT(ind == m_w_map.size());
}

CDynamicObjectArray* CFactorGraphModel::get_factor_types() const
{
	SG_REF(m_factor_types);
	return m_factor_types;
}

CFactorType* CFactorGraphModel::get_factor_type(const int32_t ftype_id) const
{
	for (int32_t fi = 0; fi < m_factor_types->get_num_elements(); ++fi)
	{
		CFactorType* ftype = dynamic_cast<CFactorType*>(m_factor_types->get_element(fi));
		if (ftype_id == ftype->get_type_id())
			return ftype;

		SG_UNREF(ftype);
	}

	return NULL;
}

SGVector<int32_t> CFactorGraphModel::get_global_params_mapping() const
{
	return m_w_map.clone();
}

SGVector<int32_t> CFactorGraphModel::get_params_mapping(const int32_t ftype_id)
{
	return m_w_map.find(ftype_id);
}

int32_t CFactorGraphModel::get_dim() const
{
	return m_w_map.size();
}

SGVector<float64_t> CFactorGraphModel::fparams_to_w()
{
	REQUIRE(m_factor_types != NULL, "%s::fparams_to_w(): no factor types!\n", get_name());

	if (m_w_cache.size() != get_dim())
		m_w_cache.resize_vector(get_dim());

	int32_t offset = 0;
	for (int32_t fi = 0; fi < m_factor_types->get_num_elements(); ++fi)
	{
		CFactorType* ftype = dynamic_cast<CFactorType*>(m_factor_types->get_element(fi));
		int32_t w_dim = ftype->get_w_dim();
		offset += w_dim;
		SGVector<float64_t> fw = ftype->get_w();
		SGVector<int32_t> fw_map = get_params_mapping(ftype->get_type_id());

		ASSERT(fw_map.size() == fw.size());

		for (int32_t wi = 0; wi < w_dim; wi++)
			m_w_cache[fw_map[wi]] = fw[wi];

		SG_UNREF(ftype);
	}

	ASSERT(offset == m_w_cache.size());

	return m_w_cache;
}

void CFactorGraphModel::w_to_fparams(SGVector<float64_t> w)
{
	// if nothing changed
	if (m_w_cache.equals(w))
		return;

	if (m_verbose)
		SG_SPRINT("****** update m_w_cache!\n");

	ASSERT(w.size() == m_w_cache.size());
	m_w_cache = w.clone();

	int32_t offset = 0;
	for (int32_t fi = 0; fi < m_factor_types->get_num_elements(); ++fi)
	{
		CFactorType* ftype = dynamic_cast<CFactorType*>(m_factor_types->get_element(fi));
		int32_t w_dim = ftype->get_w_dim();
		offset += w_dim;
		SGVector<float64_t> fw(w_dim);
		SGVector<int32_t> fw_map = get_params_mapping(ftype->get_type_id());

		for (int32_t wi = 0; wi < w_dim; wi++)
			fw[wi] = m_w_cache[fw_map[wi]];

		ftype->set_w(fw);
		SG_UNREF(ftype);
	}

	ASSERT(offset == m_w_cache.size());
}

SGVector< float64_t > CFactorGraphModel::get_joint_feature_vector(int32_t feat_idx, CStructuredData* y)
{
	// factor graph instance
	CFactorGraphFeatures* mf = CFactorGraphFeatures::obtain_from_generic(m_features);
	CFactorGraph* fg = mf->get_sample(feat_idx);

	// ground truth states
	CFactorGraphObservation* fg_states = CFactorGraphObservation::obtain_from_generic(y);
	SGVector<int32_t> states = fg_states->get_data();

	// initialize psi
	SGVector<float64_t> psi(get_dim());
	psi.zero();

	// construct unnormalized psi
	CDynamicObjectArray* facs = fg->get_factors();
	for (int32_t fi = 0; fi < facs->get_num_elements(); ++fi)
	{
		CFactor* fac = dynamic_cast<CFactor*>(facs->get_element(fi));
		CTableFactorType* ftype = fac->get_factor_type();
		int32_t id = ftype->get_type_id();
		SGVector<int32_t> w_map = get_params_mapping(id);

		ASSERT(w_map.size() == ftype->get_w_dim());

		SGVector<float64_t> dat = fac->get_data();
		int32_t dat_size = dat.size();

		ASSERT(w_map.size() == dat_size * ftype->get_num_assignments());

		int32_t ei = ftype->index_from_universe_assignment(states, fac->get_variables());
		for (int32_t di = 0; di < dat_size; di++)
			psi[w_map[ei*dat_size + di]] += dat[di];

		SG_UNREF(ftype);
		SG_UNREF(fac);
	}

	// negation (-E(x,y) = <w,phi(x,y)>)
	psi.scale(-1.0);

	SG_UNREF(facs);
	SG_UNREF(fg);

	return psi;
}

// E(x_i, y; w) - E(x_i, y_i; w) >= L(y_i, y) - xi_i
// xi_i >= max oracle
// max oracle := argmax_y { L(y_i, y) - E(x_i, y; w) + E(x_i, y_i; w) }
//			:= argmin_y { -L(y_i, y) + E(x_i, y; w) } - E(x_i, y_i; w)
// we do energy minimization in inference, so get back to max oracle value is:
// [ L(y_i, y_star) - E(x_i, y_star; w) ] + E(x_i, y_i; w)
CResultSet* CFactorGraphModel::argmax(SGVector<float64_t> w, int32_t feat_idx, bool const training)
{
	// factor graph instance
	CFactorGraphFeatures* mf = CFactorGraphFeatures::obtain_from_generic(m_features);
	CFactorGraph* fg = mf->get_sample(feat_idx);

	// prepare factor graph
	fg->connect_components();
	if (m_inf_type == TREE_MAX_PROD)
	{
		ASSERT(fg->is_tree_graph());
	}

	if (m_verbose)
		SG_SPRINT("\n------ example %d\n", feat_idx);

	// update factor parameters
	w_to_fparams(w);
	fg->compute_energies();

	if (m_verbose)
	{
		SG_SPRINT("energy table before loss-aug:\n");
		fg->evaluate_energies();
	}

	// prepare CResultSet
	CResultSet* ret = new CResultSet();
	SG_REF(ret);

	// y_truth
	CFactorGraphObservation* y_truth =
		CFactorGraphObservation::obtain_from_generic(m_labels->get_label(feat_idx));

	SGVector<int32_t> states_gt = y_truth->get_data();

	// E(x_i, y_i; w)
	float64_t energy_gt = fg->evaluate_energy(states_gt);
	ret->score = energy_gt;

	// - min_y [ E(x_i, y; w) - delta(y_i, y) ]
	CMAPInference infer_met(fg, m_inf_type);
	if (training)
	{
		fg->loss_augmentation(y_truth); // wrong assignments -delta()

		if (m_verbose)
		{
			SG_SPRINT("energy table after loss-aug:\n");
			fg->evaluate_energies();
		}
	}

	infer_met.inference();

	// y_star
	CFactorGraphObservation* y_star = infer_met.get_structured_outputs();
	SGVector<int32_t> states_star = y_star->get_data();

	ret->argmax = y_star;
	float64_t l_energy_pred = fg->evaluate_energy(states_star);
	ret->score -= l_energy_pred;
	ret->delta = delta_loss(y_truth, y_star);

	if (m_verbose)
	{
		SGVector<float64_t> psi_pred = get_joint_feature_vector(feat_idx, y_star);
		SGVector<float64_t> psi_truth = get_joint_feature_vector(feat_idx, y_truth);

		float64_t dot_pred = SGVector< float64_t >::dot(w.vector, psi_pred.vector, w.vlen);
		float64_t dot_truth = SGVector< float64_t >::dot(w.vector, psi_truth.vector, w.vlen);
		float64_t slack =  dot_pred + ret->delta - dot_truth;

		SG_SPRINT("\n");
		w.display_vector("w");

		psi_pred.display_vector("psi_pred");
		states_star.display_vector("state_pred");

		SG_SPRINT("dot_pred = %f, energy_pred = %f, delta = %f\n\n", dot_pred, l_energy_pred, ret->delta);

		psi_truth.display_vector("psi_truth");
		states_gt.display_vector("state_gt");

		SG_SPRINT("dot_truth = %f, energy_gt = %f\n\n", dot_truth, energy_gt);

		SG_SPRINT("slack = %f, score = %f\n\n", slack, ret->score);
	}

	SG_UNREF(y_truth);
	SG_UNREF(fg);

	return ret;
}

float64_t CFactorGraphModel::delta_loss(CStructuredData* y1, CStructuredData* y2)
{
	CFactorGraphObservation* y_truth = CFactorGraphObservation::obtain_from_generic(y1);
	CFactorGraphObservation* y_pred = CFactorGraphObservation::obtain_from_generic(y2);
	SGVector<int32_t> s_truth = y_truth->get_data();
	SGVector<int32_t> s_pred = y_pred->get_data();

	ASSERT(s_pred.size() == s_truth.size());

	float64_t loss = 0.0;
	for (int32_t si = 0; si < s_pred.size(); si++)
	{
		if (s_pred[si] != s_truth[si])
			loss += y_truth->get_loss_weights()[si];
	}

	return loss;
}

void CFactorGraphModel::init_training()
{
}

void CFactorGraphModel::init_primal_opt(
		float64_t regularization,
		SGMatrix< float64_t > & A,
		SGVector< float64_t > a,
		SGMatrix< float64_t > B,
		SGVector< float64_t > & b,
		SGVector< float64_t > lb,
		SGVector< float64_t > ub,
		SGMatrix< float64_t > & C)
{
	C = SGMatrix< float64_t >::create_identity_matrix(get_dim(), regularization);
}
