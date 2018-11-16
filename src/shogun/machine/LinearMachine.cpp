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

CLinearMachine::CLinearMachine(): CMachine()
{
	init();
}

CLinearMachine::CLinearMachine(CLinearMachine* machine) : CMachine()
{
	init();
	REQUIRE(machine, "No machine provided.\n");

	auto w = machine->get_w();
	auto w_clone = w.clone();
	set_w(w_clone);
	set_bias(machine->get_bias());
}

void CLinearMachine::init()
{
	bias = 0;
	features = NULL;

	SG_ADD(&m_w, "w", "Parameter vector w.");
	SG_ADD(&bias, "bias", "Bias b.");
	SG_ADD(
	    (CFeatures**)&features, "features", "Feature object.");
}


CLinearMachine::~CLinearMachine()
{
	SG_UNREF(features);
}

float64_t CLinearMachine::apply_one(int32_t vec_idx)
{
	return features->dense_dot(vec_idx, m_w.vector, m_w.vlen) + bias;
}

CRegressionLabels* CLinearMachine::apply_regression(CFeatures* data)
{
	SGVector<float64_t> outputs = apply_get_outputs(data);
	return new CRegressionLabels(outputs);
}

CBinaryLabels* CLinearMachine::apply_binary(CFeatures* data)
{
	SGVector<float64_t> outputs = apply_get_outputs(data);
	return new CBinaryLabels(outputs);
}

SGVector<float64_t> CLinearMachine::apply_get_outputs(CFeatures* data)
{
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n")

		set_features((CDotFeatures*) data);
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

SGVector<float64_t> CLinearMachine::get_w() const
{
	return m_w;
}

void CLinearMachine::set_w(const SGVector<float64_t> w)
{
	m_w = w;
}

void CLinearMachine::set_bias(float64_t b)
{
	bias=b;
}

float64_t CLinearMachine::get_bias()
{
	return bias;
}

void CLinearMachine::set_features(CDotFeatures* feat)
{
	SG_REF(feat);
	SG_UNREF(features);
	features=feat;
}

CDotFeatures* CLinearMachine::get_features()
{
	SG_REF(features);
	return features;
}

void CLinearMachine::store_model_features()
{
}
