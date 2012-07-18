/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/classifier/L1LogisticRegression.h>
#include <shogun/lib/slep/slep_logistic.h>
#include <shogun/lib/slep/slep_options.h>

namespace shogun
{

CL1LogisticRegression::CL1LogisticRegression() :
	CSLEPMachine() 
{
}

CL1LogisticRegression::CL1LogisticRegression(
     float64_t z, CDotFeatures* train_features, 
     CBinaryLabels* train_labels) :
	CSLEPMachine(z,train_features,(CLabels*)train_labels)
{
}

CL1LogisticRegression::~CL1LogisticRegression()
{
}

bool CL1LogisticRegression::train_machine(CFeatures* data)
{
	if (data && (CDotFeatures*)data)
		set_features((CDotFeatures*)data);

	ASSERT(features);
	ASSERT(m_labels);

	int32_t n_vecs = m_labels->get_num_labels();
	SGVector<float64_t> y(n_vecs);
	for (int32_t i=0; i<n_vecs; i++)
		y[i] = ((CBinaryLabels*)m_labels)->get_label(i);
	
	slep_options options = slep_options::default_options();
	options.mode = PLAIN;
	options.regularization = m_regularization;
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;
	options.rsL2 = 0.0;

	slep_result_t result = slep_solver(features, y.vector, m_z, options);

	int32_t n_feats = features->get_dim_feature_space();
	SGVector<float64_t> new_w(n_feats);
	for (int i=0; i<n_feats; i++)
		new_w[i] = result.w[i];

	set_bias(result.c[0]);

	w = new_w;
		
	return true;
}

float64_t CL1LogisticRegression::apply_one(int32_t vec_idx)
{
	return CMath::exp(-(features->dense_dot(vec_idx, w.vector, w.vlen) + bias));
}

SGVector<float64_t> CL1LogisticRegression::apply_get_outputs(CFeatures* data)
{
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n");

		set_features((CDotFeatures*) data);
	}

	if (!features)
		return SGVector<float64_t>();

	int32_t num=features->get_num_vectors();
	ASSERT(num>0);
	ASSERT(w.vlen==features->get_dim_feature_space());

	float64_t* out=SG_MALLOC(float64_t, num);
	features->dense_dot_range(out, 0, num, NULL, w.vector, w.vlen, bias);
	for (int32_t i=0; i<num; i++)
		out[i] = 2.0/(1.0+CMath::exp(-out[i])) - 1.0;//*CMath::exp(-CMath::sign(out[i])*out[i]);
	return SGVector<float64_t>(out,num);
}

}
