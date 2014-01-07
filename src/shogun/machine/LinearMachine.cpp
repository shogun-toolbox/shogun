/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <machine/LinearMachine.h>
#include <labels/RegressionLabels.h>
#include <base/Parameter.h>

using namespace shogun;

CLinearMachine::CLinearMachine()
: CMachine(), bias(0), features(NULL)
{
	init();
}

CLinearMachine::CLinearMachine(CLinearMachine* machine) : CMachine(),
	bias(0), features(NULL)
{
	set_w(machine->get_w().clone());
	set_bias(machine->get_bias());

	init();
}

void CLinearMachine::init()
{
	SG_ADD(&w, "w", "Parameter vector w.", MS_NOT_AVAILABLE);
	SG_ADD(&bias, "bias", "Bias b.", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &features, "features", "Feature object.",
	    MS_NOT_AVAILABLE);
}


CLinearMachine::~CLinearMachine()
{
	SG_UNREF(features);
}

float64_t CLinearMachine::apply_one(int32_t vec_idx)
{
	return features->dense_dot(vec_idx, w.vector, w.vlen) + bias;
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
	ASSERT(w.vlen==features->get_dim_feature_space())

	float64_t* out=SG_MALLOC(float64_t, num);
	features->dense_dot_range(out, 0, num, NULL, w.vector, w.vlen, bias);
	return SGVector<float64_t>(out,num);
}

SGVector<float64_t> CLinearMachine::get_w() const
{
	return w;
}

void CLinearMachine::set_w(const SGVector<float64_t> src_w)
{
	w=src_w;
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

