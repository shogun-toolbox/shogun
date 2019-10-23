/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evgeniy Andreev, Evan Shelhamer
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

#include <shogun/distributions/LinearHMM.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

LinearHMM::LinearHMM() : Distribution()
{
	init();
}

LinearHMM::LinearHMM(const std::shared_ptr<StringFeatures<uint16_t>>& f)
: Distribution()
{
	init();

	set_features(f);
}

void LinearHMM::set_features(std::shared_ptr<Features> f)
{
	auto string_feats = f->as<StringFeatures<uint16_t>>();
	require(string_feats, "LinearHMM works with string features.");

	Distribution::set_features(f);

	sequence_length = string_feats->get_vector_length(0);
	num_symbols     = (int32_t) string_feats->get_num_symbols();
	num_params      = sequence_length*num_symbols;
}

LinearHMM::LinearHMM(int32_t p_num_features, int32_t p_num_symbols)
: Distribution()
{
	init();

	sequence_length = p_num_features;
	num_symbols     = p_num_symbols;
	num_params      = sequence_length*num_symbols;
}

LinearHMM::~LinearHMM()
{
}

bool LinearHMM::train(std::shared_ptr<Features> data)
{
	if (data)
	{
		if (data->get_feature_class() != C_STRING ||
				data->get_feature_type() != F_WORD)
		{
			error("Expected features of class string type word!");
		}
		set_features(data);
	}
	SGMatrix<int32_t> int_transition_probs(num_symbols, sequence_length);

	int32_t vec;
	int32_t i;

	auto sf = std::static_pointer_cast<StringFeatures<uint16_t>>(features);
	for (vec=0; vec<features->get_num_vectors(); vec++)
	{
		int32_t len;
		bool free_vec;

		uint16_t* vector=sf->get_feature_vector(vec, len, free_vec);

		//just count the symbols per position -> transition_probsogram
		for (int32_t feat=0; feat<len ; feat++)
			int_transition_probs[feat*num_symbols+vector[feat]]++;

		sf->free_feature_vector(vector, vec, free_vec);
	}

	//trade memory for speed
	transition_probs = SGMatrix<float64_t>(num_symbols, sequence_length);
	log_transition_probs = SGMatrix<float64_t>(num_symbols, sequence_length);

	for (i=0;i<sequence_length;i++)
	{
		for (int32_t j=0; j<num_symbols; j++)
		{
			float64_t sum=0;
			int32_t offs=i*num_symbols+
				sf->get_masked_symbols((uint16_t)j,(uint8_t) 254);
			int32_t original_num_symbols=(int32_t)
				sf->get_original_num_symbols();

			for (int32_t k=0; k<original_num_symbols; k++)
				sum+=int_transition_probs[offs+k];

			transition_probs[i*num_symbols+j]=
				(int_transition_probs[i*num_symbols+j]+pseudo_count)/
				(sum+sf->get_original_num_symbols()*pseudo_count);
			log_transition_probs[i*num_symbols+j]=
				log(transition_probs[i*num_symbols+j]);
		}
	}

	return true;
}

bool LinearHMM::train(
	const int32_t* indizes, int32_t num_indizes, float64_t pseudo)
{
	SGMatrix<int32_t> int_transition_probs(num_symbols, sequence_length);
	int32_t vec;
	int32_t i;

	auto sf = std::static_pointer_cast<StringFeatures<uint16_t>>(features);
	for (vec=0; vec<num_indizes; vec++)
	{
		int32_t len;
		bool free_vec;

		ASSERT(indizes[vec]>=0 &&
			indizes[vec]<sf->get_num_vectors());
		uint16_t* vector=sf->get_feature_vector(indizes[vec], len, free_vec);
		sf->free_feature_vector(vector, indizes[vec], free_vec);

		//just count the symbols per position -> transition_probsogram
		//
		for (int32_t feat=0; feat<len ; feat++)
			int_transition_probs[feat*num_symbols+vector[feat]]++;
	}

	//trade memory for speed
	transition_probs = SGMatrix<float64_t>(num_symbols, sequence_length);
	log_transition_probs = SGMatrix<float64_t>(num_symbols, sequence_length);

	for (i=0;i<sequence_length;i++)
	{
		for (int32_t j=0; j<num_symbols; j++)
		{
			float64_t sum=0;
			int32_t original_num_symbols=(int32_t)
				sf->get_original_num_symbols();
			for (int32_t k=0; k<original_num_symbols; k++)
			{
				sum+=int_transition_probs[i*num_symbols+
					sf->get_masked_symbols((uint16_t)j,(uint8_t) 254)+k];
			}

			transition_probs[i*num_symbols+j]=
				(int_transition_probs[i*num_symbols+j]+pseudo)/
				(sum+sf->get_original_num_symbols()*pseudo);
			log_transition_probs[i*num_symbols+j]=
				log(transition_probs[i*num_symbols+j]);
		}
	}

	return true;
}

float64_t LinearHMM::get_log_likelihood_example(uint16_t* vector, int32_t len)
{
	float64_t result=log_transition_probs[vector[0]];

	for (int32_t i=1; i<len; i++)
		result+=log_transition_probs[i*num_symbols+vector[i]];

	return result;
}

float64_t LinearHMM::get_log_likelihood_example(int32_t num_example)
{
	int32_t len;
	bool free_vec;
	auto sf = std::static_pointer_cast<StringFeatures<uint16_t>>(features);
	uint16_t* vector=sf->get_feature_vector(num_example, len, free_vec);
	float64_t result=get_log_likelihood_example(vector, len);

	sf->free_feature_vector(vector, num_example, free_vec);

	return result;
}

float64_t LinearHMM::get_likelihood_example(uint16_t* vector, int32_t len)
{
	float64_t result=transition_probs[vector[0]];

	for (int32_t i=1; i<len; i++)
		result*=transition_probs[i*num_symbols+vector[i]];

	return result;
}

float64_t LinearHMM::get_likelihood_example(int32_t num_example)
{
	int32_t len;
	bool free_vec;
	auto sf = std::static_pointer_cast<StringFeatures<uint16_t>>(features);
	uint16_t* vector=sf->get_feature_vector(num_example, len, free_vec);

	float64_t result=get_likelihood_example(vector, len);

	sf->free_feature_vector(vector, num_example, free_vec);

	return result;
}

float64_t LinearHMM::get_log_derivative(int32_t num_param, int32_t num_example)
{
	int32_t len;
	bool free_vec;
	auto sf = std::static_pointer_cast<StringFeatures<uint16_t>>(features);
	uint16_t* vector=sf->get_feature_vector(num_example, len, free_vec);
	float64_t result=0;
	int32_t position=num_param/num_symbols;
	ASSERT(position>=0 && position<len)
	uint16_t sym=(uint16_t) (num_param-position*num_symbols);

	if (vector[position]==sym && transition_probs[num_param]!=0)
		result=1.0/transition_probs[num_param];
	sf->free_feature_vector(vector, num_example, free_vec);

	return result;
}

SGMatrix<float64_t> LinearHMM::get_transition_probs()
{
	return transition_probs;
}

bool LinearHMM::set_transition_probs(const SGMatrix<float64_t>& probs)
{
	require(
		probs.num_rows == num_symbols && probs.num_cols == sequence_length,
		"Transition matrix should have a dimension of (%d, %d).", num_symbols, sequence_length);

	if (log_transition_probs.num_rows != num_symbols || log_transition_probs.num_cols != sequence_length)
		log_transition_probs = SGMatrix<float64_t>(num_symbols, sequence_length);

	if (transition_probs.num_rows != num_symbols || transition_probs.num_cols != sequence_length)
		transition_probs = SGMatrix<float64_t>(num_symbols, sequence_length);

	for (int32_t i=0; i<num_params; i++)
	{
		transition_probs[i]=probs[i];
		log_transition_probs[i]=log(transition_probs[i]);
	}

	return true;
}

SGMatrix<float64_t> LinearHMM::get_log_transition_probs()
{
	return log_transition_probs;
}

bool LinearHMM::set_log_transition_probs(const SGMatrix<float64_t>& probs)
{
	require(
		probs.num_rows == num_symbols && probs.num_cols == sequence_length,
		"Transition matrix log should have a dimension of (%d, %d).", num_symbols, sequence_length);

	if (log_transition_probs.num_rows != num_symbols || log_transition_probs.num_cols != sequence_length)
		log_transition_probs = SGMatrix<float64_t>(num_symbols, sequence_length);

	if (transition_probs.num_rows != num_symbols || transition_probs.num_cols != sequence_length)
		transition_probs = SGMatrix<float64_t>(num_symbols, sequence_length);

	for (int32_t i=0; i<num_params; i++)
	{
		log_transition_probs[i]=probs[i];
		transition_probs[i]=exp(log_transition_probs[i]);
	}

	return true;
}

void LinearHMM::load_serializable_post() noexcept(false)
{
	SGObject::load_serializable_post();

	num_params = sequence_length*num_symbols;
}

void LinearHMM::init()
{
	sequence_length = 0;
	num_symbols = 0;
	num_params = 0;
	transition_probs = SGMatrix<float64_t>();
	log_transition_probs = SGMatrix<float64_t>();

	SG_ADD(&transition_probs, "transition_probs", "Transition probabilities.");
	SG_ADD(&log_transition_probs, "log_transition_probs", "Transition probabilities (logspace).");
}
