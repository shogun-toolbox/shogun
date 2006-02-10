#include "distributions/Distribution.h"
#include "lib/Mathmatics.h"

CDistribution::CDistribution() : features(NULL), pseudo_count(0)
{
}


CDistribution::~CDistribution()
{
}

REAL CDistribution::get_log_likelihood_sample()
{
	ASSERT(features);

	REAL sum=0;
	for (INT i=0; i<features->get_num_vectors(); i++)
		sum+=get_log_likelihood_example(i);

	return sum/features->get_num_vectors();
}

REAL* CDistribution::get_log_likelihood_all_examples()
{
	ASSERT(features);

	REAL* output=new REAL[features->get_num_vectors()];
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
