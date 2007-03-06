/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "distributions/Distribution.h"
#include "lib/Mathematics.h"

CDistribution::CDistribution() : CSGObject(), features(NULL), pseudo_count(0)
{
}


CDistribution::~CDistribution()
{
}

DREAL CDistribution::get_log_likelihood_sample()
{
	ASSERT(features);

	DREAL sum=0;
	for (INT i=0; i<features->get_num_vectors(); i++)
		sum+=get_log_likelihood_example(i);

	return sum/features->get_num_vectors();
}

DREAL* CDistribution::get_log_likelihood_all_examples()
{
	ASSERT(features);

	DREAL* output=new DREAL[features->get_num_vectors()];
	ASSERT(output);

	for (INT i=0; i<features->get_num_vectors(); i++)
		output[i]=get_log_likelihood_example(i);

	return output;
}

INT CDistribution::get_num_relevant_model_parameters()
{
	INT total_num=get_num_model_parameters();
	INT num=0;

	for (INT i=0; i<total_num; i++)
	{
		if (get_log_model_parameter(i)>CMath::ALMOST_NEG_INFTY)
			num++;
	}
	return num;
}
