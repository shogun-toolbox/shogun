/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evgeniy Andreev, Evan Shelhamer, Sergey Lisitsyn
 */

#include <shogun/distributions/Histogram.h>
#include <shogun/lib/common.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

Histogram::Histogram()
: Distribution()
{
	hist=SG_CALLOC(float64_t, 1<<16);
}

Histogram::Histogram(std::shared_ptr<StringFeatures<uint16_t>> f)
: Distribution()
{
	hist=SG_CALLOC(float64_t, 1<<16);
	features=f;
}

Histogram::~Histogram()
{
	SG_FREE(hist);
}

bool Histogram::train(std::shared_ptr<Features> data)
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

	auto sf = std::static_pointer_cast<StringFeatures<uint16_t>>(features);
	for (vec=0; vec<features->get_num_vectors(); vec++)
	{
		int32_t len;
		bool free_vec;

		uint16_t* vector=sf->get_feature_vector(vec, len, free_vec);

		for (feat=0; feat<len ; feat++)
			hist[vector[feat]]++;

		sf->free_feature_vector(vector, vec, free_vec);
	}

	for (i=0; i< (int32_t) (1<<16); i++)
		hist[i]=log(hist[i]);

	return true;
}

float64_t Histogram::get_log_likelihood_example(int32_t num_example)
{
	ASSERT(features)
	ASSERT(features->get_feature_class()==C_STRING)
	ASSERT(features->get_feature_type()==F_WORD)

	int32_t len;
	bool free_vec;
	float64_t loglik=0;

	auto sf = std::static_pointer_cast<StringFeatures<uint16_t>>(features);
	uint16_t* vector=sf->get_feature_vector(num_example, len, free_vec);

	for (int32_t i=0; i<len; i++)
		loglik+=hist[vector[i]];

	sf->free_feature_vector(vector, num_example, free_vec);

	return loglik;
}

float64_t Histogram::get_log_derivative(int32_t num_param, int32_t num_example)
{
	if (hist[num_param] < Math::ALMOST_NEG_INFTY)
		return -Math::INFTY;
	else
	{
		ASSERT(features)
		ASSERT(features->get_feature_class()==C_STRING)
		ASSERT(features->get_feature_type()==F_WORD)

		int32_t len;
		bool free_vec;
		float64_t deriv=0;

		auto sf = std::static_pointer_cast<StringFeatures<uint16_t>>(features);
		uint16_t* vector=sf->get_feature_vector(num_example, len, free_vec);

		int32_t num_occurences=0;

		for (int32_t i=0; i<len; i++)
		{
			deriv+=hist[vector[i]];

			if (vector[i]==num_param)
				num_occurences++;
		}

		sf->free_feature_vector(vector, num_example, free_vec);

		if (num_occurences>0)
			deriv += std::log((float64_t)num_occurences) - hist[num_param];
		else
			deriv=-Math::INFTY;

		return deriv;
	}
}

float64_t Histogram::get_log_model_parameter(int32_t num_param)
{
	return hist[num_param];
}

bool Histogram::set_histogram(const SGVector<float64_t> histogram)
{
	ASSERT(histogram.vlen==get_num_model_parameters())

	SG_FREE(hist);
	hist=SG_MALLOC(float64_t, histogram.vlen);
	for (int32_t i=0; i<histogram.vlen; i++)
		hist[i]=histogram.vector[i];

	return true;
}

SGVector<float64_t> Histogram::get_histogram()
{
	return SGVector<float64_t>(hist,get_num_model_parameters(),false);
}
