/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2010 Soeren Sonnenburg
 * Written (W) 2011 Shashwat Lal Das
 * Modifications (W) 2013 Thoralf Klein
 * Copyright (c) 2007-2009 The LIBLINEAR Project.
 * Copyright (C) 2007-2010 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <classifier/svm/OnlineLibLinear.h>
#include <features/streaming/StreamingDenseFeatures.h>
#include <features/streaming/StreamingSparseFeatures.h>
#include <lib/Time.h>

using namespace shogun;

COnlineLibLinear::COnlineLibLinear()
		: COnlineLinearMachine()
{
		init();
}

COnlineLibLinear::COnlineLibLinear(float64_t C_reg)
{
		init();
		C1=C_reg;
		C2=C_reg;
		use_bias=true;
}

COnlineLibLinear::COnlineLibLinear(
		float64_t C_reg, CStreamingDotFeatures* traindat)
{
		init();
		C1=C_reg;
		C2=C_reg;
		use_bias=true;

		set_features(traindat);
}

COnlineLibLinear::COnlineLibLinear(COnlineLibLinear *mch)
{
	init();
	C1 = mch->C1;
	C2 = mch->C2;
	use_bias = mch->use_bias;

	set_features(mch->features);

	w_dim = mch->w_dim;
	if (w_dim > 0)
	{
		w = SG_MALLOC(float32_t, w_dim);
		memcpy(w, mch->w, w_dim*sizeof(float32_t));
	}
	else
	{
		w = NULL;
	}
	bias = mch->bias;
}


void COnlineLibLinear::init()
{
		C1=1;
		C2=1;
		use_bias=false;

		m_parameters->add(&C1, "C1",  "C Cost constant 1.");
		m_parameters->add(&C2, "C2",  "C Cost constant 2.");
		m_parameters->add(&use_bias, "use_bias",  "Indicates if bias is used.");
}

COnlineLibLinear::~COnlineLibLinear()
{
}

void COnlineLibLinear::start_train()
{
	Cp = C1;
	Cn = C2;
	PGmax_old = CMath::INFTY;
	PGmin_old = -CMath::INFTY;
	PGmax_new = -CMath::INFTY;
	PGmin_new = CMath::INFTY;

	diag[0]=0;diag[1]=0;diag[2]=0;
	upper_bound[0]=Cn;upper_bound[1]=0;upper_bound[2]=Cp;

	bias = 0;

	PGmax_new = -CMath::INFTY;
	PGmin_new = CMath::INFTY;

	v = 0;
	nSV = 0;
}

void COnlineLibLinear::stop_train()
{
	float64_t gap = PGmax_new - PGmin_new;

	SG_DONE()
	SG_INFO("Optimization finished.\n")

	// calculate objective value
	for (int32_t i=0; i<w_dim; i++)
		v += w[i]*w[i];
	v += bias*bias;

	SG_INFO("Objective value = %lf\n", v/2)
	SG_INFO("nSV = %d\n", nSV)
	SG_INFO("gap = %g\n", gap)
}

void COnlineLibLinear::train_one(SGVector<float32_t> ex, float64_t label)
{
	alpha_current = 0;
	if (label > 0)
		y_current = +1;
	else
		y_current = -1;

	QD = diag[y_current + 1];
	// Dot product of vector with itself
	QD += SGVector<float32_t>::dot(ex.vector, ex.vector, ex.vlen);

	// Dot product of vector with learned weights
	G = SGVector<float32_t>::dot(ex.vector, w, w_dim);

	if (use_bias)
		G += bias;
	G = G*y_current - 1;
	// LINEAR TERM PART?

	C = upper_bound[y_current + 1];
	G += alpha_current*diag[y_current + 1]; // Can be eliminated, since diag = 0 vector

	PG = 0;
	if (alpha_current == 0) // This condition will always be true in the online version
	{
		if (G > PGmax_old)
		{
			return;
		}
		else if (G < 0)
			PG = G;
	}
	else if (alpha_current == C)
	{
		if (G < PGmin_old)
		{
			return;
		}
		else if (G > 0)
			PG = G;
	}
	else
		PG = G;

	PGmax_new = CMath::max(PGmax_new, PG);
	PGmin_new = CMath::min(PGmin_new, PG);

	if (fabs(PG) > 1.0e-12)
	{
		float64_t alpha_old = alpha_current;
		alpha_current = CMath::min(CMath::max(alpha_current - G/QD, 0.0), C);
		d = (alpha_current - alpha_old) * y_current;

		for (int32_t i=0; i < w_dim; ++i)
			w[i] += d*ex[i];


		if (use_bias)
			bias += d;
	}

	v += alpha_current*(alpha_current*diag[y_current + 1] - 2);
	if (alpha_current > 0)
		nSV++;
}

void COnlineLibLinear::train_one(SGSparseVector<float32_t> ex, float64_t label)
{
	alpha_current = 0;
	if (label > 0)
		y_current = +1;
	else
		y_current = -1;

	QD = diag[y_current + 1];
	// Dot product of vector with itself
	QD += SGSparseVector<float32_t>::sparse_dot(ex, ex);

	// Dot product of vector with learned weights
	G = ex.dense_dot(1.0,w,w_dim,0.0);

	if (use_bias)
		G += bias;
	G = G*y_current - 1;
	// LINEAR TERM PART?

	C = upper_bound[y_current + 1];
	G += alpha_current*diag[y_current + 1]; // Can be eliminated, since diag = 0 vector

	PG = 0;
	if (alpha_current == 0) // This condition will always be true in the online version
	{
		if (G > PGmax_old)
		{
			return;
		}
		else if (G < 0)
			PG = G;
	}
	else if (alpha_current == C)
	{
		if (G < PGmin_old)
		{
			return;
		}
		else if (G > 0)
			PG = G;
	}
	else
		PG = G;

	PGmax_new = CMath::max(PGmax_new, PG);
	PGmin_new = CMath::min(PGmin_new, PG);

	if (fabs(PG) > 1.0e-12)
	{
		float64_t alpha_old = alpha_current;
		alpha_current = CMath::min(CMath::max(alpha_current - G/QD, 0.0), C);
		d = (alpha_current - alpha_old) * y_current;

		for (int32_t i=0; i < ex.num_feat_entries; i++)
			w[ex.features[i].feat_index] += d*ex.features[i].entry;


		if (use_bias)
			bias += d;
	}

	v += alpha_current*(alpha_current*diag[y_current + 1] - 2);
	if (alpha_current > 0)
		nSV++;
}

void COnlineLibLinear::train_example(CStreamingDotFeatures *feature, float64_t label)
{
	features->expand_if_required(w, w_dim);

	if (features->get_feature_class() == C_STREAMING_DENSE) {
		CStreamingDenseFeatures<float32_t> *feat =
			dynamic_cast<CStreamingDenseFeatures<float32_t> *>(feature);
		if (feat == NULL)
			SG_ERROR("Expected streaming dense feature <float32_t>\n")

		train_one(feat->get_vector(), label);
	}
	else if (features->get_feature_class() == C_STREAMING_SPARSE) {
		CStreamingSparseFeatures<float32_t> *feat =
			dynamic_cast<CStreamingSparseFeatures<float32_t> *>(feature);
		if (feat == NULL)
			SG_ERROR("Expected streaming sparse feature <float32_t>\n")

		train_one(feat->get_vector(), label);
	}
	else {
		SG_NOTIMPLEMENTED
	}
}
