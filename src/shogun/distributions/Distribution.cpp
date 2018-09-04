/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg
 */

#include <shogun/distributions/Distribution.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CDistribution::CDistribution()
: CSGObject(), features(NULL), pseudo_count(1e-10)
{
}


CDistribution::~CDistribution()
{
	SG_UNREF(features);
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

float64_t CDistribution::update_params_em(const SGVector<float64_t> alpha_k)
{
	SG_WARNING("Not implemented in this class. This class cannot be used for Mixture models.\n")
	SG_NOTIMPLEMENTED
	return -1;
}

CDistribution* CDistribution::obtain_from_generic(CSGObject* object)
{
	if (!object)
		return NULL;

	CDistribution* casted=dynamic_cast<CDistribution*>(object);
	if (!casted)
		return NULL;

	return casted;
}