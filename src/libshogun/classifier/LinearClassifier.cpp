/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/LinearClassifier.h"
#include "base/Parameter.h"

using namespace shogun;

CLinearClassifier::CLinearClassifier()
: CClassifier(), w_dim(0), w(NULL), bias(0), features(NULL)
{

	m_parameters->add_vector(&w, &w_dim, "w", "Parameter vector w.");
	m_parameters->add(&bias, "bias", "Bias b.");
	m_parameters->add((CSGObject**) &features, "features", "Feature object.");

}

CLinearClassifier::~CLinearClassifier()
{
	delete[] w;
	SG_UNREF(features);
}

bool CLinearClassifier::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool CLinearClassifier::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

CLabels* CLinearClassifier::classify()
{
	if (features)
	{
		int32_t num=features->get_num_vectors();
		ASSERT(num>0);
		ASSERT(w_dim==features->get_dim_feature_space());

		float64_t* out=new float64_t[num];
		features->dense_dot_range(out, 0, num, NULL, w, w_dim, bias);

		CLabels* output=new CLabels(num);
		output->set_labels(out, num);

		delete[] out;

		return output;
	}

	return NULL;
}

CLabels* CLinearClassifier::classify(CFeatures* data)
{
	if (!data)
		SG_ERROR("No features specified\n");
	if (!data->has_property(FP_DOT))
		SG_ERROR("Specified features are not of type CDotFeatures\n");
	set_features((CDotFeatures*) data);
	return classify();
}
