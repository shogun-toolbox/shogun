/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Heiko Strathmann,
 *          Evan Shelhamer
 */

#include <shogun/classifier/PluginEstimate.h>
#include <shogun/distributions/LinearHMM.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/common.h>

using namespace shogun;

PluginEstimate::PluginEstimate(float64_t pos_pseudo, float64_t neg_pseudo)
: Machine(), m_pos_pseudo(1e-10), m_neg_pseudo(1e-10),
	pos_model(NULL), neg_model(NULL), features(NULL)
{
	SG_ADD(
	    &m_pos_pseudo, "pos_pseudo", "pseudo count for positive class",
	    ParameterProperties::MODEL);
	SG_ADD(
	    &m_neg_pseudo, "neg_pseudo", "pseudo count for negative class",
	    ParameterProperties::MODEL);
	SG_ADD(
	    &pos_model, "pos_model", "LinearHMM modelling positive class.",
	    ParameterProperties::MODEL);
	SG_ADD(
	    &neg_model, "neg_model", "LinearHMM modelling negative class.",
	    ParameterProperties::MODEL);
	SG_ADD(&features, "features", "String Features.");
}

PluginEstimate::~PluginEstimate()
{
}

bool PluginEstimate::train_machine(const std::shared_ptr<Features>& data, const std::shared_ptr<Labels>& labs)
{
	if (data)
	{
		if (data->get_feature_class() != C_STRING ||
		    data->get_feature_type() != F_WORD)
		{
			error("Features not of class string type word");
		}

		set_features(std::static_pointer_cast<StringFeatures<uint16_t>>(data));
	}
	ASSERT(features)

	pos_model=std::make_shared<LinearHMM>(features);
	neg_model=std::make_shared<LinearHMM>(features);

	int32_t* pos_indizes=SG_MALLOC(int32_t, std::static_pointer_cast<StringFeatures<uint16_t>>(features)->get_num_vectors());
	int32_t* neg_indizes=SG_MALLOC(int32_t, std::static_pointer_cast<StringFeatures<uint16_t>>(features)->get_num_vectors());

	ASSERT(labs->get_num_labels() == features->get_num_vectors())

	int32_t pos_idx = 0;
	int32_t neg_idx = 0;

	auto binary_labels = std::static_pointer_cast<BinaryLabels>(labs);
	for (int32_t i=0; i<labs->get_num_labels(); i++)
	{
		if (binary_labels->get_label(i) > 0)
			pos_indizes[pos_idx++]=i;
		else
			neg_indizes[neg_idx++] = i;
	}

	io::info("training using pseudos {} and {}", m_pos_pseudo, m_neg_pseudo);
	pos_model->train(pos_indizes, pos_idx, m_pos_pseudo);
	neg_model->train(neg_indizes, neg_idx, m_neg_pseudo);

	SG_FREE(pos_indizes);
	SG_FREE(neg_indizes);

	return true;
}

std::shared_ptr<BinaryLabels> PluginEstimate::apply_binary(std::shared_ptr<Features> data)
{
	if (data)
	{
		if (data->get_feature_class() != C_STRING ||
		    data->get_feature_type() != F_WORD)
		{
			error("Features not of class string type word");
		}

		set_features(std::dynamic_pointer_cast<StringFeatures<uint16_t>>(data));
	}

	ASSERT(features)
	SGVector<float64_t> result(features->get_num_vectors());

	for (int32_t vec = 0; vec < features->get_num_vectors(); vec++)
		result[vec] = apply_one(vec);

	return std::make_shared<BinaryLabels>(result);
}

float64_t PluginEstimate::apply_one(int32_t vec_idx)
{
	ASSERT(features)

	int32_t len;
	bool free_vec;
	uint16_t* vector = features->get_feature_vector(vec_idx, len, free_vec);

	if ((!pos_model) || (!neg_model))
		error("model(s) not assigned");

	float64_t result = pos_model->get_log_likelihood_example(vector, len) -
	                   neg_model->get_log_likelihood_example(vector, len);
	features->free_feature_vector(vector, vec_idx, free_vec);
	return result;
}
