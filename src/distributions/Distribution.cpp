#include "distributions/Distribution.h"
#include "lib/Mathmatics.h"

#include <assert.h>

CDistribution::CDistribution() : features(NULL), pseudo_count(0)
{
}


CDistribution::~CDistribution()
{
}

REAL CDistribution::get_log_likelihood_sample()
{
	assert(features);

	REAL sum=0;
	for (INT i=0; i<features->get_num_vectors(); i++)
		sum+=get_log_likelihood_example(i);

	return sum/features->get_num_vectors();
}

REAL* CDistribution::get_log_likelihood_all_examples()
{
	assert(features);

	REAL* output=new REAL[features->get_num_vectors()];
	assert(output);

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
		if (get_log_model_parameter(i)>math.ALMOST_NEG_INFTY)
			num++;
	}
	return num;
}
