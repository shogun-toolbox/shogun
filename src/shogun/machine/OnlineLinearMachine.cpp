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

using namespace shogun;

COnlineLinearMachine::COnlineLinearMachine()
: CMachine(), w_dim(0), w(NULL), bias(0), features(NULL)
{
	m_parameters->add_vector(&w, &w_dim, "w", "Parameter vector w.");
	SG_ADD(&bias, "bias", "Bias b.", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &features, "features",
	    "Feature object.", MS_NOT_AVAILABLE);
}

COnlineLinearMachine::~COnlineLinearMachine()
{
	// It is possible that a derived class may have already
	// called SG_FREE() on the weight vector
	if (w != NULL)
		SG_FREE(w);
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

	DynArray<float64_t>* labels_dynarray=new DynArray<float64_t>();
	int32_t num_labels=0;

	features->start_parser();
	while (features->get_next_example())
	{
		float64_t current_lab=features->dense_dot(w, w_dim) + bias;

		labels_dynarray->append_element(current_lab);
		num_labels++;

		features->release_example();
	}
	features->end_parser();

	SGVector<float64_t> labels_array(num_labels);
	for (int32_t i=0; i<num_labels; i++)
		labels_array.vector[i]=(*labels_dynarray)[i];

	delete labels_dynarray;
	return labels_array;
}

float32_t COnlineLinearMachine::apply_one(float32_t* vec, int32_t len)
{
		return SGVector<float32_t>::dot(vec, w, len)+bias;
}

float32_t COnlineLinearMachine::apply_to_current_example()
{
		return features->dense_dot(w, w_dim)+bias;
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
