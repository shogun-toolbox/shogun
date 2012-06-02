
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Soeren Sonnenburg
 */

#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/regression/svr/LibLinearRegression.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>

using namespace shogun;

CLibLinearRegression::CLibLinearRegression() :
	CLinearMachine()
{
	init_defaults();
}

CLibLinearRegression::CLibLinearRegression(float64_t C, CDotFeatures* feats, CLabels* labs) :
	CLinearMachine()
{
	init_defaults();
	set_C(C);
	set_features(feats);
	set_labels(labs);
}

void CLibLinearRegression::init_defaults()
{
	set_C(1.0);
	set_epsilon(1e-2);
	set_max_iter(10000);
	set_use_bias(false);
}

void CLibLinearRegression::register_parameters()
{
	SG_ADD(&m_C, "m_C", "regularization constant",MS_AVAILABLE);
	SG_ADD(&m_epsilon, "m_epsilon", "tolerance epsilon",MS_NOT_AVAILABLE);
	SG_ADD(&m_epsilon, "m_tube_epsilon", "svr tube epsilon",MS_AVAILABLE);
	SG_ADD(&m_max_iter, "m_max_iter", "max number of iterations",MS_NOT_AVAILABLE);
	SG_ADD(&m_use_bias, "m_use_bias", "indicates whether bias should be used",MS_NOT_AVAILABLE);
}

CLibLinearRegression::~CLibLinearRegression()
{
}

bool CLibLinearRegression::train_machine(CFeatures* data)
{
	if (data)
		set_features((CDotFeatures*)data);

	ASSERT(features);
	ASSERT(m_labels && m_labels->get_label_type()==LT_REGRESSION);
	//TODO
	return true;
}
#endif /* HAVE_LAPACK */
