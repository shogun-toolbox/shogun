/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/distributions/Histogram.h>
#include <shogun/lib/common.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CHistogram::CHistogram()
: CDistribution()
{
	hist=SG_CALLOC(float64_t, 1<<16);
}

CHistogram::CHistogram(CStringFeatures<uint16_t> *f)
: CDistribution()
{
	hist=SG_CALLOC(float64_t, 1<<16);
	features=f;
	SG_REF(features);
}

CHistogram::~CHistogram()
{
	SG_FREE(hist);
}

bool CHistogram::train(CFeatures* data)
{
	int32_t vec;
	int32_t feat;
	int32_t i;

	if (data)
	{
		if (data->get_feature_class() != C_STRING ||
				data->get_feature_type() != F_WORD)
		{
			SG_ERROR("Expected features of class string type word\n")
		}
		set_features(data);
	}

	ASSERT(features)
	ASSERT(features->get_feature_class()==C_STRING)
	ASSERT(features->get_feature_type()==F_WORD)

	for (i=0; i< (int32_t) (1<<16); i++)
		hist[i]=0;

	for (vec=0; vec<features->get_num_vectors(); vec++)
	{
		int32_t len;
		bool free_vec;

		uint16_t* vector=((CStringFeatures<uint16_t>*) features)->
			get_feature_vector(vec, len, free_vec);

		for (feat=0; feat<len ; feat++)
			hist[vector[feat]]++;

		((CStringFeatures<uint16_t>*) features)->
			free_feature_vector(vector, vec, free_vec);
	}

	for (i=0; i< (int32_t) (1<<16); i++)
		hist[i]=log(hist[i]);

	return true;
}

float64_t CHistogram::get_log_likelihood_example(int32_t num_example)
{
	ASSERT(features)
	ASSERT(features->get_feature_class()==C_STRING)
	ASSERT(features->get_feature_type()==F_WORD)

	int32_t len;
	bool free_vec;
	float64_t loglik=0;

	uint16_t* vector=((CStringFeatures<uint16_t>*) features)->
		get_feature_vector(num_example, len, free_vec);

	for (int32_t i=0; i<len; i++)
		loglik+=hist[vector[i]];

	((CStringFeatures<uint16_t>*) features)->
		free_feature_vector(vector, num_example, free_vec);

	return loglik;
}

float64_t CHistogram::get_log_derivative(int32_t num_param, int32_t num_example)
{
	if (hist[num_param] < CMath::ALMOST_NEG_INFTY)
		return -CMath::INFTY;
	else
	{
		ASSERT(features)
		ASSERT(features->get_feature_class()==C_STRING)
		ASSERT(features->get_feature_type()==F_WORD)

		int32_t len;
		bool free_vec;
		float64_t deriv=0;

		uint16_t* vector=((CStringFeatures<uint16_t>*) features)->
			get_feature_vector(num_example, len, free_vec);

		int32_t num_occurences=0;

		for (int32_t i=0; i<len; i++)
		{
			deriv+=hist[vector[i]];

			if (vector[i]==num_param)
				num_occurences++;
		}

		((CStringFeatures<uint16_t>*) features)->
			free_feature_vector(vector, num_example, free_vec);

		if (num_occurences>0)
			deriv+=CMath::log((float64_t) num_occurences)-hist[num_param];
		else
			deriv=-CMath::INFTY;

		return deriv;
	}
}

float64_t CHistogram::get_log_model_parameter(int32_t num_param)
{
	return hist[num_param];
}

bool CHistogram::set_histogram(const SGVector<float64_t> histogram)
{
	ASSERT(histogram.vlen==get_num_model_parameters())

	SG_FREE(hist);
	hist=SG_MALLOC(float64_t, histogram.vlen);
	for (int32_t i=0; i<histogram.vlen; i++)
		hist[i]=histogram.vector[i];

	return true;
}

SGVector<float64_t> CHistogram::get_histogram()
{
	return SGVector<float64_t>(hist,get_num_model_parameters(),false);
}

