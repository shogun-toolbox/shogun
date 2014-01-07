/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2009-2011 Jun Liu, Jieping Ye
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <mathematics/SparseInverseCovariance.h>
#include <base/Parameter.h>
#include <lib/slep/SpInvCoVa/invCov.h>

using namespace shogun;

CSparseInverseCovariance::CSparseInverseCovariance() :
	CSGObject(), m_lasso_max_iter(1000),
	m_max_iter(1000), m_f_gap(1e-6), m_x_gap(1e-4),
	m_xtol(1e-4)
{
	register_parameters();
}

CSparseInverseCovariance::~CSparseInverseCovariance()
{
}

void CSparseInverseCovariance::register_parameters()
{
	SG_ADD(&m_lasso_max_iter,"lasso_max_iter",
	       "maximum iteration of LASSO step",MS_NOT_AVAILABLE);
	SG_ADD(&m_max_iter,"max_iter","maximum total iteration",
	       MS_NOT_AVAILABLE);
	SG_ADD(&m_f_gap,"f_gap","f gap",MS_NOT_AVAILABLE);
	SG_ADD(&m_x_gap,"x_gap","x gap",MS_NOT_AVAILABLE);
	SG_ADD(&m_xtol,"xtol","xtol",MS_NOT_AVAILABLE);
}

SGMatrix<float64_t> CSparseInverseCovariance::estimate(SGMatrix<float64_t> S, float64_t lambda_c)
{
	ASSERT(S.num_cols==S.num_rows)

	int32_t n = S.num_cols;
	float64_t sum_S = 0.0;
	for (int32_t i=0; i<n; i++)
		sum_S += S(i,i);

	float64_t* Theta = SG_CALLOC(float64_t, n*n);
	float64_t* W = SG_CALLOC(float64_t, n*n);

	invCov(Theta, W, S.matrix, lambda_c, sum_S, n, m_lasso_max_iter,
	       m_f_gap, m_x_gap, m_max_iter, m_xtol);

	SG_FREE(W);
	return SGMatrix<float64_t>(Theta,n,n);
}
