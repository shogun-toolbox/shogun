/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/FeatureTreeLeastSquaresRegression.h>
#include <shogun/lib/slep/slep_tree_lsr.h>
#include <shogun/lib/slep/slep_options.h>

namespace shogun
{

CFeatureTreeLeastSquaresRegression::CFeatureTreeLeastSquaresRegression() :
	CLinearMachine(), m_z(1.0), m_feature_tree(NULL)
{
}

CFeatureTreeLeastSquaresRegression::CFeatureTreeLeastSquaresRegression(
     float64_t z, CDotFeatures* train_features, 
     CLabels* train_labels, CFeatureTree* feature_tree) :
	CLinearMachine(), m_z(1.0), m_feature_tree(NULL)
{
	set_z(z);
	set_features(train_features);
	set_labels(train_labels);
	SG_SPRINT("FT=%d\n",m_feature_tree);
	set_feature_tree(feature_tree);
	set_termination(slep_options::get_default_termination());
	set_regularization(slep_options::get_default_regularization());
	set_tolerance(slep_options::get_default_tolerance());
	set_max_iter(slep_options::get_default_max_iter());
}

CFeatureTreeLeastSquaresRegression::~CFeatureTreeLeastSquaresRegression()
{
	SG_UNREF(m_feature_tree);
}

void CFeatureTreeLeastSquaresRegression::register_parameters()
{
	SG_ADD(&m_z, "z", "regularization coefficient", MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_feature_tree, "feature_tree", "feature tree", MS_NOT_AVAILABLE);
	SG_ADD(&m_termination, "termination", "termination", MS_NOT_AVAILABLE);
	SG_ADD(&m_regularization, "regularization", "regularization", MS_NOT_AVAILABLE);
	SG_ADD(&m_tolerance, "tolerance", "tolerance", MS_NOT_AVAILABLE);
	SG_ADD(&m_max_iter, "max_iter", "maximum number of iterations", MS_NOT_AVAILABLE);
}

CFeatureTree* CFeatureTreeLeastSquaresRegression::get_feature_tree() const
{
	SG_REF(m_feature_tree);
	return m_feature_tree;
}

int32_t CFeatureTreeLeastSquaresRegression::get_max_iter() const
{
	return m_max_iter;
}
int32_t CFeatureTreeLeastSquaresRegression::get_regularization() const
{
	return m_regularization;
}
int32_t CFeatureTreeLeastSquaresRegression::get_termination() const
{
	return m_termination;
}
float64_t CFeatureTreeLeastSquaresRegression::get_tolerance() const
{
	return m_tolerance;
}
float64_t CFeatureTreeLeastSquaresRegression::get_z() const
{
	return m_z;
}

void CFeatureTreeLeastSquaresRegression::set_feature_tree(CFeatureTree* feature_tree)
{
	SG_UNREF(m_feature_tree);
	SG_REF(feature_tree);
	m_feature_tree = feature_tree;
}

void CFeatureTreeLeastSquaresRegression::set_max_iter(int32_t max_iter)
{
	m_max_iter = max_iter;
}
void CFeatureTreeLeastSquaresRegression::set_regularization(int32_t regularization)
{
	m_regularization = regularization;
}
void CFeatureTreeLeastSquaresRegression::set_termination(int32_t termination)
{
	m_termination = termination;
}
void CFeatureTreeLeastSquaresRegression::set_tolerance(float64_t tolerance)
{
	m_tolerance = tolerance;
}
void CFeatureTreeLeastSquaresRegression::set_z(float64_t z)
{
	m_z = z;
}

bool CFeatureTreeLeastSquaresRegression::train_machine(CFeatures* data)
{
	if (data && (CDotFeatures*)data)
		set_features((CDotFeatures*)data);

	ASSERT(features);
	ASSERT(m_labels);

	int32_t n_vectors = features->get_num_vectors();

	float64_t* y = SG_MALLOC(float64_t, n_vectors);
	for (int32_t i=0; i<n_vectors; i++)
		y[i] = m_labels->get_label(i);

	slep_options options;
	options.general = false;
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;
	options.restart_num = 100;
	options.n_nodes = 1;
	options.regularization = 0;
	options.ind = SG_CALLOC(double,3);
	options.ind[0] = -1; 
	options.ind[1] = -1; 
	options.ind[2] = 1.0;
	options.G = NULL;
	options.initial_w = NULL;

	SG_FREE(w);
	w = slep_tree_lsr(features,y,m_z,options);
	w_dim = features->get_dim_feature_space();

	SG_FREE(y);

	return true;
}

}
