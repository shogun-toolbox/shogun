/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Evan Shelhamer,
 *          Youssef Emad El-Din, Evgeniy Andreev, Thoralf Klein, Bjoern Esser,
 *          Fernando Iglesias
 */

#include <rxcpp/rx-lite.hpp>
#include <shogun/features/DotFeatures.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/machine/LinearMachine.h>

using namespace shogun;

LinearMachine::LinearMachine(): Machine()
{
	init();
}

LinearMachine::LinearMachine(std::shared_ptr<LinearMachine> machine) : Machine()
{
	init();
	REQUIRE(machine, "No machine provided.\n");

	auto w = machine->get_w();
	auto w_clone = w.clone();
	set_w(w_clone);
	set_bias(machine->get_bias());
}

void LinearMachine::init()
{
	bias = 0;
	features = NULL;

	SG_ADD(&m_w, "w", "Parameter vector w.", ParameterProperties::MODEL);
	SG_ADD(&bias, "bias", "Bias b.", ParameterProperties::MODEL);
	SG_ADD(
	    (std::shared_ptr<Features>*)&features, "features", "Feature object.");
}


LinearMachine::~LinearMachine()
{

}

float64_t LinearMachine::apply_one(int32_t vec_idx)
{
	return features->dense_dot(vec_idx, m_w.vector, m_w.vlen) + bias;
}

std::shared_ptr<RegressionLabels> LinearMachine::apply_regression(std::shared_ptr<Features> data)
{
	SGVector<float64_t> outputs = apply_get_outputs(data);
	return std::make_shared<RegressionLabels>(outputs);
}

std::shared_ptr<BinaryLabels> LinearMachine::apply_binary(std::shared_ptr<Features> data)
{
	SGVector<float64_t> outputs = apply_get_outputs(data);
	return std::make_shared<BinaryLabels>(outputs);
}

SGVector<float64_t> LinearMachine::apply_get_outputs(std::shared_ptr<Features> data)
{
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type DotFeatures\n")

		set_features(std::static_pointer_cast<DotFeatures>(data));
	}

	if (!features)
		return SGVector<float64_t>();

	int32_t num=features->get_num_vectors();
	ASSERT(num>0)
	ASSERT(m_w.vlen==features->get_dim_feature_space())

	float64_t* out=SG_MALLOC(float64_t, num);
	features->dense_dot_range(out, 0, num, NULL, m_w.vector, m_w.vlen, bias);
	return SGVector<float64_t>(out,num);
}

SGVector<float64_t> LinearMachine::get_w() const
{
	return m_w;
}

void LinearMachine::set_w(const SGVector<float64_t> w)
{
	m_w = w;
}

void LinearMachine::set_bias(float64_t b)
{
	bias=b;
}

float64_t LinearMachine::get_bias() const
{
	return bias;
}

void LinearMachine::set_features(std::shared_ptr<DotFeatures> feat)
{


	features=feat;
}

std::shared_ptr<DotFeatures> LinearMachine::get_features()
{

	return features;
}

void LinearMachine::store_model_features()
{
}
