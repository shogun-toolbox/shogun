/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#include <shogun/regression/gp/InferenceMethod.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <iostream>

using namespace shogun;

CInferenceMethod::CInferenceMethod() {
	kernel = NULL;
	m_model = NULL;
	m_labels = NULL;
	features = NULL;
	mean = NULL;
}

CInferenceMethod::CInferenceMethod(CKernel* kern, CDotFeatures* feat,
		CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod)
{
	set_kernel(kern);
	set_features(feat);
	set_labels(lab);
	set_model(mod);
	set_mean(m);
}

CInferenceMethod::~CInferenceMethod() {
	SG_UNREF(kernel);
	SG_UNREF(features);
	SG_UNREF(m_labels);
	SG_UNREF(m_model);

}




