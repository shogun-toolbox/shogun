/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/machine/LinearMachine.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

CLinearMachine::CLinearMachine()
: CMachine(), w_dim(0), w(NULL), bias(0), features(NULL)
{
	init();
}

CLinearMachine::CLinearMachine(CLinearMachine* machine) : CMachine(),
	w_dim(0), w(NULL), bias(0), features(NULL)
{
	set_w(machine->get_w().clone());
	set_bias(machine->get_bias());

	init();
}

void CLinearMachine::init()
{
	m_parameters->add_vector(&w, &w_dim, "w", "Parameter vector w.");
	SG_ADD(&bias, "bias", "Bias b.", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &features, "features", "Feature object.",
	    MS_NOT_AVAILABLE);
}


CLinearMachine::~CLinearMachine()
{
	SG_FREE(w);
	SG_UNREF(features);
}

bool CLinearMachine::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool CLinearMachine::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

CLabels* CLinearMachine::apply()
{
	if (!features)
		return NULL;

	int32_t num=features->get_num_vectors();
	ASSERT(num>0);
	ASSERT(w_dim==features->get_dim_feature_space());

	float64_t* out=SG_MALLOC(float64_t, num);
	features->dense_dot_range(out, 0, num, NULL, w, w_dim, bias);

	return new CLabels(SGVector<float64_t>(out,num));
}

CLabels* CLinearMachine::apply(CFeatures* data)
{
	if (!data)
		SG_ERROR("No features specified\n");
	if (!data->has_property(FP_DOT))
		SG_ERROR("Specified features are not of type CDotFeatures\n");
	set_features((CDotFeatures*) data);
	return apply();
}
