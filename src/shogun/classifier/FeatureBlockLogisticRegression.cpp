/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/classifier/FeatureBlockLogisticRegression.h>
#include <shogun/lib/slep/slep_solver.h>
#include <shogun/lib/slep/slep_options.h>

#include <shogun/lib/IndexBlockGroup.h>
#include <shogun/lib/IndexBlockTree.h>

namespace shogun
{

CFeatureBlockLogisticRegression::CFeatureBlockLogisticRegression() :
	CLinearMachine()
{
	init();
	register_parameters();
}

CFeatureBlockLogisticRegression::CFeatureBlockLogisticRegression(
     float64_t z, CDotFeatures* train_features,
     CBinaryLabels* train_labels, CIndexBlockRelation* feature_relation) :
	CLinearMachine()
{
	init();
	set_feature_relation(feature_relation);
	set_z(z);
	set_features(train_features);
	set_labels(train_labels);
	register_parameters();
}

void CFeatureBlockLogisticRegression::init()
{
	m_feature_relation=NULL;
	m_z=0.0;
	m_q=2.0;
	m_termination=0;
	m_regularization=0;
	m_tolerance=1e-3;
	m_max_iter=1000;
}

CFeatureBlockLogisticRegression::~CFeatureBlockLogisticRegression()
{
	SG_UNREF(m_feature_relation);
}

void CFeatureBlockLogisticRegression::register_parameters()
{
	SG_ADD((CSGObject**)&m_feature_relation, "feature_relation", "feature relation", MS_NOT_AVAILABLE);
	SG_ADD(&m_z, "z", "regularization coefficient", MS_AVAILABLE);
	SG_ADD(&m_q, "q", "q of L1/Lq", MS_AVAILABLE);
	SG_ADD(&m_termination, "termination", "termination", MS_NOT_AVAILABLE);
	SG_ADD(&m_regularization, "regularization", "regularization", MS_NOT_AVAILABLE);
	SG_ADD(&m_tolerance, "tolerance", "tolerance", MS_NOT_AVAILABLE);
	SG_ADD(&m_max_iter, "max_iter", "maximum number of iterations", MS_NOT_AVAILABLE);
}

CIndexBlockRelation* CFeatureBlockLogisticRegression::get_feature_relation() const
{
	SG_REF(m_feature_relation);
	return m_feature_relation;
}

void CFeatureBlockLogisticRegression::set_feature_relation(CIndexBlockRelation* feature_relation)
{
	SG_REF(feature_relation);
	SG_UNREF(m_feature_relation);
	m_feature_relation = feature_relation;
}

int32_t CFeatureBlockLogisticRegression::get_max_iter() const
{
	return m_max_iter;
}

int32_t CFeatureBlockLogisticRegression::get_regularization() const
{
	return m_regularization;
}

int32_t CFeatureBlockLogisticRegression::get_termination() const
{
	return m_termination;
}

float64_t CFeatureBlockLogisticRegression::get_tolerance() const
{
	return m_tolerance;
}

float64_t CFeatureBlockLogisticRegression::get_z() const
{
	return m_z;
}

float64_t CFeatureBlockLogisticRegression::get_q() const
{
	return m_q;
}

void CFeatureBlockLogisticRegression::set_max_iter(int32_t max_iter)
{
	ASSERT(max_iter>=0)
	m_max_iter = max_iter;
}

void CFeatureBlockLogisticRegression::set_regularization(int32_t regularization)
{
	ASSERT(regularization==0 || regularization==1)
	m_regularization = regularization;
}

void CFeatureBlockLogisticRegression::set_termination(int32_t termination)
{
	ASSERT(termination>=0 && termination<=4)
	m_termination = termination;
}

void CFeatureBlockLogisticRegression::set_tolerance(float64_t tolerance)
{
	ASSERT(tolerance>0.0)
	m_tolerance = tolerance;
}

void CFeatureBlockLogisticRegression::set_z(float64_t z)
{
	m_z = z;
}

void CFeatureBlockLogisticRegression::set_q(float64_t q)
{
	m_q = q;
}

bool CFeatureBlockLogisticRegression::train_machine(CFeatures* data)
{
	if (data && (CDotFeatures*)data)
		set_features((CDotFeatures*)data);

	ASSERT(features)
	ASSERT(m_labels)

	int32_t n_vecs = m_labels->get_num_labels();
	SGVector<float64_t> y(n_vecs);
	for (int32_t i=0; i<n_vecs; i++)
		y[i] = ((CBinaryLabels*)m_labels)->get_label(i);

	slep_options options = slep_options::default_options();
	options.q = m_q;
	options.regularization = m_regularization;
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;
	options.loss = LOGISTIC;

	EIndexBlockRelationType relation_type = m_feature_relation->get_relation_type();
	switch (relation_type)
	{
		case GROUP:
		{
			CIndexBlockGroup* feature_group = (CIndexBlockGroup*)m_feature_relation;
			SGVector<index_t> ind = feature_group->get_SLEP_ind();
			options.ind = ind.vector;
			options.n_feature_blocks = ind.vlen-1;
			if (ind[ind.vlen-1] > features->get_dim_feature_space())
				SG_ERROR("Group of features covers more features than available\n")

			options.gWeight = SG_MALLOC(double, options.n_feature_blocks);
			for (int32_t i=0; i<options.n_feature_blocks; i++)
				options.gWeight[i] = 1.0;
			options.mode = FEATURE_GROUP;
			options.loss = LOGISTIC;
			options.n_nodes = 0;
			slep_result_t result = slep_solver(features, y.vector, m_z, options);

			SG_FREE(options.gWeight);
			int32_t n_feats = features->get_dim_feature_space();
			SGVector<float64_t> new_w(n_feats);
			for (int i=0; i<n_feats; i++)
				new_w[i] = result.w[i];
			set_bias(result.c[0]);

			w = new_w;
		}
		break;
		case TREE:
		{
			CIndexBlockTree* feature_tree = (CIndexBlockTree*)m_feature_relation;

			SGVector<float64_t> ind_t = feature_tree->get_SLEP_ind_t();
			SGVector<float64_t> G;
			if (feature_tree->is_general())
			{
				G = feature_tree->get_SLEP_G();
				options.general = true;
			}
			options.ind_t = ind_t.vector;
			options.G = G.vector;
			options.n_nodes = ind_t.vlen/3;
			options.n_feature_blocks = ind_t.vlen/3;
			options.mode = FEATURE_TREE;
			options.loss = LOGISTIC;

			slep_result_t result = slep_solver(features, y.vector, m_z, options);

			int32_t n_feats = features->get_dim_feature_space();
			SGVector<float64_t> new_w(n_feats);
			for (int i=0; i<n_feats; i++)
				new_w[i] = result.w[i];

			set_bias(result.c[0]);

			w = new_w;
		}
		break;
		default:
			SG_ERROR("Not supported feature relation type\n")
	}

	return true;
}

float64_t CFeatureBlockLogisticRegression::apply_one(int32_t vec_idx)
{
	return CMath::exp(-(features->dense_dot(vec_idx, w.vector, w.vlen) + bias));
}

SGVector<float64_t> CFeatureBlockLogisticRegression::apply_get_outputs(CFeatures* data)
{
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n")

		set_features((CDotFeatures*) data);
	}

	if (!features)
		return SGVector<float64_t>();

	int32_t num=features->get_num_vectors();
	ASSERT(num>0)
	ASSERT(w.vlen==features->get_dim_feature_space())

	float64_t* out=SG_MALLOC(float64_t, num);
	features->dense_dot_range(out, 0, num, NULL, w.vector, w.vlen, bias);
	for (int32_t i=0; i<num; i++)
		out[i] = 2.0/(1.0+CMath::exp(-out[i])) - 1.0;
	return SGVector<float64_t>(out,num);
}

}
