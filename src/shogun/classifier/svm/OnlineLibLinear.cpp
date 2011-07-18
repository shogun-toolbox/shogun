/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2010 Soeren Sonnenburg
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (c) 2007-2009 The LIBLINEAR Project.
 * Copyright (C) 2007-2010 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/classifier/svm/OnlineLibLinear.h>

using namespace shogun;

COnlineLibLinear::COnlineLibLinear()
		: COnlineLinearMachine()
{
		init();
}

COnlineLibLinear::COnlineLibLinear(float64_t C)
{
		init();
		C1=C;
		C2=C;
		use_bias=true;
}

COnlineLibLinear::COnlineLibLinear(
		float64_t C, CStreamingDotFeatures* traindat)
{
		init();
		C1=C;
		C2=C;
		use_bias=true;

		set_features(traindat);
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

bool COnlineLibLinear::train(CFeatures* data)
{
		if (data)
		{
				if (!data->has_property(FP_STREAMING_DOT))
						SG_ERROR("Specified features are not of type CStreamingDotFeatures\n");
				set_features((CStreamingDotFeatures*) data);
		}
		
		float64_t C, d, G;
		float64_t QD;

		// y and alpha for example being processed
		int32_t y_current;
		float64_t alpha_current;

		// Cost constants
		float64_t Cp=C1;
		float64_t Cn=C2;

		// PG: projected gradient, for shrinking and stopping
		float64_t PG;
		float64_t PGmax_old = CMath::INFTY;
		float64_t PGmin_old = -CMath::INFTY;
		float64_t PGmax_new = -CMath::INFTY;
		float64_t PGmin_new = CMath::INFTY;

		// Diag is probably unnecessary
		float64_t diag[3] = {0, 0, 0};
		float64_t upper_bound[3] = {Cn, 0, Cp};

		// Bias
		bias = 0;

		PGmax_new = -CMath::INFTY;
		PGmin_new = CMath::INFTY;

		// Objective value = v/2
		float64_t v = 0;
		// Number of support vectors
		int32_t nSV = 0;

		// Start reading the examples
		features->start_parser();

		CTime start_time;
		while (features->get_next_example())
		{
				alpha_current = 0;
				if (features->get_label() > 0)
						y_current = +1;
				else
						y_current = -1;

				QD = diag[y_current + 1];
				// Dot product of vector with itself
				QD += features->dot(features);

				features->expand_if_required(w, w_dim);

				G = features->dense_dot(w, w_dim);
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
								features->release_example();
								continue;
						}
						else if (G < 0)
								PG = G;
				}
				else if (alpha_current == C)
				{
						if (G < PGmin_old)
						{
								features->release_example();
								continue;
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

						features->add_to_dense_vec(d, w, w_dim);

						if (use_bias)
								bias += d;
				}

				v += alpha_current*(alpha_current*diag[y_current + 1] - 2);
				if (alpha_current > 0)
						nSV++;

				features->release_example();
		}

		features->end_parser();

		float64_t gap = PGmax_new - PGmin_new;

		SG_DONE();
		SG_INFO("Optimization finished.\n");

		// calculate objective value
		for (int32_t i=0; i<w_dim; i++)
				v += w[i]*w[i];
		v += bias*bias;

		SG_INFO("Objective value = %lf\n", v/2);
		SG_INFO("nSV = %d\n", nSV);

		return true;
}
