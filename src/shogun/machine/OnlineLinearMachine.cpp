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
	m_parameters->add(&bias, "bias", "Bias b.");
	m_parameters->add((CSGObject**) &features, "features", "Feature object.");
}

COnlineLinearMachine::~COnlineLinearMachine()
{
	SG_FREE(w);
	SG_UNREF(features);
}

bool COnlineLinearMachine::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool COnlineLinearMachine::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

CLabels* COnlineLinearMachine::apply()
{
	ASSERT(features);
	ASSERT(features->has_property(FP_STREAMING_DOT));

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

	return new CLabels(labels_array);
}

CLabels* COnlineLinearMachine::apply(CFeatures* data)
{
	if (!data)
		SG_ERROR("No features specified\n");
	if (!data->has_property(FP_STREAMING_DOT))
		SG_ERROR("Specified features are not of type CStreamingDotFeatures\n");
	set_features((CStreamingDotFeatures*) data);
	return apply();
}

float64_t COnlineLinearMachine::apply(float64_t* vec, int32_t len)
{
		return CMath::dot(vec, w, len)+bias;
}

float64_t COnlineLinearMachine::apply_to_current_example()
{
		return features->dense_dot(w, w_dim)+bias;
}
