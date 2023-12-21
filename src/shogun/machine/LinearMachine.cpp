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
#include <utility>

using namespace shogun;

LinearMachine::LinearMachine(): Machine()
{
	init();
}

LinearMachine::LinearMachine(const std::shared_ptr<LinearMachine>& machine) : Machine()
{
	init();
	require(machine, "No machine provided.");

	auto w = machine->get_w();
	auto w_clone = w.clone();
	set_w(w_clone);
	set_bias(machine->get_bias());
}

void LinearMachine::init()
{
	SG_ADD(&m_w, "w", "Parameter vector w.", ParameterProperties::MODEL);
	SG_ADD(&bias, "bias", "Bias b.", ParameterProperties::MODEL);
}


LinearMachine::~LinearMachine()
{

}

float64_t LinearMachine::apply_one(
    const std::shared_ptr<DotFeatures>& features, int32_t vec_idx)
{
	return features->dot(vec_idx, m_w) + bias;
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
	const auto features = data->as<DotFeatures>();
	int32_t num=features->get_num_vectors();
	require(
	    m_w.vlen == features->get_dim_feature_space(),
	    "Fetures expected to have {} dimentions", m_w.vlen);
	SGVector<float64_t> out(num);
	features->dense_dot_range(out.vector, 0, num, NULL, m_w.vector, m_w.vlen, bias);
	return out;
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


