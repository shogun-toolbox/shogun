/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 * Copyright (C) 2013 Roman Votyakov
 */

#include <shogun/machine/gp/InferenceMethod.h>
#include <shogun/features/CombinedFeatures.h>

using namespace shogun;

CInferenceMethod::CInferenceMethod()
{
	init();
}

CInferenceMethod::CInferenceMethod(CKernel* kern, CFeatures* feat,
		CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod)
{
	init();

	set_kernel(kern);
	set_features(feat);
	set_labels(lab);
	set_model(mod);
	set_mean(m);
}

CInferenceMethod::~CInferenceMethod()
{
	SG_UNREF(m_kernel);
	SG_UNREF(m_features);
	SG_UNREF(m_labels);
	SG_UNREF(m_model);
	SG_UNREF(m_mean);
}

void CInferenceMethod::init()
{
	SG_ADD((CSGObject**)&m_kernel, "kernel", "Kernel", MS_AVAILABLE);
	SG_ADD(&m_scale, "scale", "Kernel Scale", MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_model, "likelihood_model", "Likelihood model",
			MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_labels, "labels", "Labels", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_features, "features", "Features", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_mean, "mean_function", "Mean Function", MS_NOT_AVAILABLE);

	m_kernel=NULL;
	m_model=NULL;
	m_labels=NULL;
	m_features=NULL;
	m_mean=NULL;
	m_scale=1.0;
}

void CInferenceMethod::check_members()
{
	REQUIRE(m_features, "Training features should not be NULL\n")
	REQUIRE(m_features->get_num_vectors(),
		"Number of training features must be greater than zero\n")
	REQUIRE(m_labels, "Labels should not be NULL\n")
	REQUIRE(m_labels->get_num_labels(),
			"Number of labels must be greater than zero\n")
	REQUIRE(m_labels->get_num_labels()==m_features->get_num_vectors(),
		"Number of training vectors must match number of labels, which is %d, "
		"but number of training vectors is %d\n", m_labels->get_num_labels(),
		m_features->get_num_vectors())
	REQUIRE(m_kernel, "Kernel should not be NULL\n")
	REQUIRE(m_mean, "Mean function should not be NULL\n")

	CFeatures* feat=m_features;

	if (m_features->get_feature_class()==C_COMBINED)
		feat=((CCombinedFeatures*)m_features)->get_first_feature_obj();
	else
		SG_REF(m_features);

	REQUIRE(feat->has_property(FP_DOT),
			"Training features must be type of CFeatures\n")
	REQUIRE(feat->get_feature_class()==C_DENSE, "Training features must be dense\n")
	REQUIRE(feat->get_feature_type()==F_DREAL, "Training features must be real\n")

	SG_UNREF(feat);
}
