/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <distributions/Distribution.h>
#include <mathematics/Math.h>

using namespace shogun;

CDistribution::CDistribution()
: CSGObject(), features(NULL), pseudo_count(1e-10)
{
}


CDistribution::~CDistribution()
{
}

float64_t CDistribution::get_log_likelihood_sample()
{
	ASSERT(features)

	float64_t sum=0;
	for (int32_t i=0; i<features->get_num_vectors(); i++)
		sum+=get_log_likelihood_example(i);

	return sum/features->get_num_vectors();
}

SGVector<float64_t> CDistribution::get_log_likelihood()
{
	ASSERT(features)

	int32_t num=features->get_num_vectors();
	float64_t* vec=SG_MALLOC(float64_t, num);

	for (int32_t i=0; i<num; i++)
		vec[i]=get_log_likelihood_example(i);

	return SGVector<float64_t>(vec,num);
}

int32_t CDistribution::get_num_relevant_model_parameters()
{
	int32_t total_num=get_num_model_parameters();
	int32_t num=0;

	for (int32_t i=0; i<total_num; i++)
	{
		if (get_log_model_parameter(i)>CMath::ALMOST_NEG_INFTY)
			num++;
	}
	return num;
}

SGVector<float64_t> CDistribution::get_likelihood_for_all_examples()
{
	ASSERT(features);
	int32_t num=features->get_num_vectors();
	ASSERT(num>0);

	SGVector<float64_t> result=SGVector<float64_t>(num);
	for (int32_t i=0; i<num; i++)
		result[i]=get_likelihood_example(i);

	return result;
}
