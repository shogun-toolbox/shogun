/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "distributions/histogram/Histogram.h"
#include "lib/common.h"
#include "features/StringFeatures.h"
#include "lib/io.h"
#include "lib/Mathematics.h"


CHistogram::CHistogram()
{
	hist=new DREAL[1<<16];
}

CHistogram::CHistogram(CStringFeatures<WORD> *f)
{
	hist=new DREAL[1<<16];
	features=f;
}

CHistogram::~CHistogram()
{
	delete[] hist;
}

bool CHistogram::train()
{
	INT vec;
	INT feat;
	INT i;

	ASSERT(features);
	ASSERT(features->get_feature_class()==C_STRING);
	ASSERT(features->get_feature_type()==F_WORD);

	for (i=0; i< (INT) (1<<16); i++)
		hist[i]=0;

	for (vec=0; vec<features->get_num_vectors(); vec++)
	{
		INT len;

		WORD* vector=((CStringFeatures<WORD>*) features)->get_feature_vector(vec, len);

		for (feat=0; feat<len ; feat++)
			hist[vector[feat]]++;
	}

	for (i=0; i< (INT) (1<<16); i++)
		hist[i]=log(hist[i]);

	return true;
}

DREAL CHistogram::get_log_likelihood_example(INT num_example)
{
	ASSERT(features);
	ASSERT(features->get_feature_class()==C_STRING);
	ASSERT(features->get_feature_type()==F_WORD);

	INT len;
	DREAL loglik=0;

	WORD* vector=((CStringFeatures<WORD>*) features)->get_feature_vector(num_example, len);

	for (INT i=0; i<len; i++)
		loglik+=hist[vector[i]];

	return loglik;
}

DREAL CHistogram::get_log_derivative(INT num_param, INT num_example)
{
	if (hist[num_param] < CMath::ALMOST_NEG_INFTY)
		return -CMath::INFTY;
	else
	{
		ASSERT(features);
		ASSERT(features->get_feature_class()==C_STRING);
		ASSERT(features->get_feature_type()==F_WORD);

		INT len;
		DREAL deriv=0;

		WORD* vector=((CStringFeatures<WORD>*) features)->get_feature_vector(num_example, len);

		INT num_occurences=0;

		for (INT i=0; i<len; i++)
		{
			deriv+=hist[vector[i]];

			if (vector[i]==num_param)
				num_occurences++;
		}

		if (num_occurences>0)
			deriv+=log(num_occurences)-hist[num_param];
		else
			deriv=-CMath::INFTY;

		return deriv;
	}
}

DREAL CHistogram::get_log_model_parameter(INT num_param)
{
	return hist[num_param];
}

bool CHistogram::set_histogram(DREAL* src, INT num)
{
	ASSERT(num==get_num_model_parameters());

	delete[] hist;
	hist=new DREAL[num];
	for (INT i=0; i<num; i++) {
		hist[i]=src[i];
	}

	return true;
}

void CHistogram::get_histogram(DREAL** dst, INT* num)
{
	*num=get_num_model_parameters();
	size_t sz=sizeof(*hist)*(*num);
	*dst=(DREAL*) malloc(sz);
	ASSERT(dst);

	memcpy(*dst, hist, sz);
}

