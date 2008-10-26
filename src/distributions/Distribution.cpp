/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "distributions/Distribution.h"
#include "lib/Mathematics.h"

CDistribution::CDistribution()
: CSGObject(), features(NULL), pseudo_count(1e-10)
{
}


CDistribution::~CDistribution()
{
}

DREAL CDistribution::get_log_likelihood_sample()
{
	ASSERT(features);

	DREAL sum=0;
	for (int32_t i=0; i<features->get_num_vectors(); i++)
		sum+=get_log_likelihood_example(i);

	return sum/features->get_num_vectors();
}

void CDistribution::get_log_likelihood(DREAL **dst, int32_t *num)
{
	ASSERT(features);

	*num=features->get_num_vectors();
	size_t sz=sizeof(DREAL)*(*num);
	*dst=(DREAL*) malloc(sz);
	ASSERT(dst);

	for (int32_t i=0; i<(*num); i++)
		*(*dst+i)=get_log_likelihood_example(i);
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
