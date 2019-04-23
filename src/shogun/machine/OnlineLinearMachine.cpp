/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Shashwat Lal Das, Sergey Lisitsyn, Thoralf Klein,
 *          Evgeniy Andreev, Chiyuan Zhang, Viktor Gal, Evan Shelhamer,
 *          Sanuj Sharma
 */

#include <shogun/machine/OnlineLinearMachine.h>
#include <shogun/base/Parameter.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

#include <vector>

using namespace shogun;

OnlineLinearMachine::OnlineLinearMachine()
: Machine(), bias(0), features(NULL)
{
	SG_ADD(&m_w, "m_w", "Parameter vector w.", ParameterProperties::MODEL);
	SG_ADD(&bias, "bias", "Bias b.", ParameterProperties::MODEL);
	SG_ADD((std::shared_ptr<SGObject>*) &features, "features",
	    "Feature object.");
}

OnlineLinearMachine::~OnlineLinearMachine()
{

}

std::shared_ptr<BinaryLabels> OnlineLinearMachine::apply_binary(std::shared_ptr<Features> data)
{
	SGVector<float64_t> outputs = apply_get_outputs(data);
	return std::make_shared<BinaryLabels>(outputs);
}

std::shared_ptr<RegressionLabels> OnlineLinearMachine::apply_regression(std::shared_ptr<Features> data)
{
	SGVector<float64_t> outputs = apply_get_outputs(data);
	return std::make_shared<RegressionLabels>(outputs);
}

SGVector<float64_t> OnlineLinearMachine::apply_get_outputs(std::shared_ptr<Features> data)
{
	if (data)
	{
		if (!data->has_property(FP_STREAMING_DOT))
			error("Specified features are not of type CStreamingDotFeatures");

		set_features(data->as<StreamingDotFeatures>());
	}

	ASSERT(features)
	ASSERT(features->has_property(FP_STREAMING_DOT))

	std::vector<float64_t> labels;
	features->start_parser();
	while (features->get_next_example())
	{
		float64_t current_lab=features->dense_dot(m_w.vector, m_w.vlen) + bias;

		labels.push_back(current_lab);
		features->release_example();
	}
	features->end_parser();

	SGVector<float64_t> labels_array(labels.size());
	sg_memcpy(labels_array.vector, labels.data(), sizeof(float64_t)*labels.size());

	return labels_array;
}

float32_t OnlineLinearMachine::apply_one(float32_t* vec, int32_t len)
{
		SGVector<float32_t> wrap(vec, len, false);
		return linalg::dot(wrap, m_w)+bias;
}

float32_t OnlineLinearMachine::apply_to_current_example()
{
		return features->dense_dot(m_w.vector, m_w.vlen)+bias;
}

bool OnlineLinearMachine::train_machine(std::shared_ptr<Features >data)
{
	if (data)
	{
		if (!data->has_property(FP_STREAMING_DOT))
			error("Specified features are not of type CStreamingDotFeatures");
		set_features(data->as<StreamingDotFeatures>());
	}
	start_train();
	features->start_parser();
	while (features->get_next_example())
	{
		train_example(features, features->get_label());
		features->release_example();
	}

	features->end_parser();
	stop_train();

	return true;
}
