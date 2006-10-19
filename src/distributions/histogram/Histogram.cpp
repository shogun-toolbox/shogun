/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "distributions/histogram/Histogram.h"
#include "lib/common.h"
#include "features/WordFeatures.h"
#include "lib/io.h"
#include "lib/Mathematics.h"


CHistogram::CHistogram()
{
	hist=new DREAL[1<<16];
	ASSERT(hist);
}

CHistogram::~CHistogram()
{
	delete[] hist;
}

	//const INT* indizes, INT num_indizes)
bool CHistogram::train()
{
	INT vec;
	INT feat;
	INT i;

	ASSERT(features);
	ASSERT(features->get_feature_class()==C_SIMPLE);
	ASSERT(features->get_feature_type()==F_WORD);

	for (i=0; i< (INT) (1<<16); i++)
		hist[i]=0;

	for (vec=0; vec<features->get_num_vectors(); vec++)
	{
		INT len;
		bool to_free;

		WORD* vector=((CWordFeatures*) features)->get_feature_vector(vec, len, to_free);

		for (feat=0; feat<len ; feat++)
		{
			hist[vector[feat]]++;
		}
		((CWordFeatures*) features)->free_feature_vector(vector, len, to_free);
	}

	for (i=0; i< (INT) (1<<16); i++)
		hist[i]=log(hist[i]);

	return true;
}

DREAL CHistogram::get_log_likelihood_example(INT num_example)
{
	ASSERT(features);
	ASSERT(features->get_feature_class()==C_SIMPLE);
	ASSERT(features->get_feature_type()==F_WORD);

	INT len;
	bool to_free;
	DREAL loglik=0;

	WORD* vector=((CWordFeatures*) features)->get_feature_vector(num_example, len, to_free);

	for (INT i=0; i<len; i++)
	{
		loglik+=hist[vector[i]];
	}

	((CWordFeatures*) features)->free_feature_vector(vector, len, to_free);

	return loglik;
}

DREAL CHistogram::get_log_derivative(INT num_example, INT param_num)
{
	if (hist[param_num] < CMath::ALMOST_NEG_INFTY)
		return -CMath::INFTY;
	else
	{
		ASSERT(features);
		ASSERT(features->get_feature_class()==C_SIMPLE);
		ASSERT(features->get_feature_type()==F_WORD);

		INT len;
		bool to_free;
		DREAL deriv=0;

		WORD* vector=((CWordFeatures*) features)->get_feature_vector(num_example, len, to_free);

		INT num_occurences=0;

		for (INT i=0; i<len; i++)
		{
			deriv+=hist[vector[i]];

			if (vector[i]==param_num)
				num_occurences++;
		}

		if (num_occurences>0)
			deriv+=log(num_occurences)-hist[param_num];
		else
			deriv=-CMath::INFTY;

		((CWordFeatures*) features)->free_feature_vector(vector, len, to_free);

		return deriv;
	}
}

DREAL CHistogram::get_log_model_parameter(INT param_num)
{
	return hist[param_num];
}
