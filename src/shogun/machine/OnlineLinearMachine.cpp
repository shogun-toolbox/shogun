/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/machine/OnlineLinearMachine.h>
#include <shogun/base/Parameter.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

#include <vector>

using namespace shogun;

COnlineLinearMachine::COnlineLinearMachine()
: CMachine(), bias(0), features(NULL)
{
	SG_ADD(&m_w, "m_w", "Parameter vector w.", MS_NOT_AVAILABLE);
	SG_ADD(&bias, "bias", "Bias b.", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &features, "features",
	    "Feature object.", MS_NOT_AVAILABLE);
}

COnlineLinearMachine::~COnlineLinearMachine()
{
	SG_UNREF(features);
}

CBinaryLabels* COnlineLinearMachine::apply_binary(CFeatures* data)
{
	SGVector<float64_t> outputs = apply_get_outputs(data);
	return new CBinaryLabels(outputs);
}

CRegressionLabels* COnlineLinearMachine::apply_regression(CFeatures* data)
{
	SGVector<float64_t> outputs = apply_get_outputs(data);
	return new CRegressionLabels(outputs);
}

SGVector<float64_t> COnlineLinearMachine::apply_get_outputs(CFeatures* data)
{
	if (data)
	{
		if (!data->has_property(FP_STREAMING_DOT))
			SG_ERROR("Specified features are not of type CStreamingDotFeatures\n")

		set_features((CStreamingDotFeatures*) data);
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

float32_t COnlineLinearMachine::apply_one(float32_t* vec, int32_t len)
{
		SGVector<float32_t> wrap(vec, len, false);
		return linalg::dot(wrap, m_w)+bias;
}

float32_t COnlineLinearMachine::apply_to_current_example()
{
		return features->dense_dot(m_w.vector, m_w.vlen)+bias;
}

bool COnlineLinearMachine::train_machine(CFeatures *data)
{
	if (data)
	{
		if (!data->has_property(FP_STREAMING_DOT))
			SG_ERROR("Specified features are not of type CStreamingDotFeatures\n")
		set_features((CStreamingDotFeatures*) data);
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
