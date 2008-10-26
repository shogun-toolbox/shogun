/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/LinearClassifier.h"

CLinearClassifier::CLinearClassifier()
: CClassifier(), w_dim(0), w(NULL), bias(0), features(NULL)
{
}

CLinearClassifier::~CLinearClassifier()
{
	delete[] w;
	SG_UNREF(features);
}

bool CLinearClassifier::load(FILE* srcfile)
{
	return false;
}

bool CLinearClassifier::save(FILE* dstfile)
{
	return false;
}

CLabels* CLinearClassifier::classify(CLabels* output)
{
	if (features)
	{
		int32_t num=features->get_num_vectors();
		ASSERT(num>0);
		ASSERT(w_dim==features->get_num_features());

		if (!output)
			output=new CLabels(num);
		ASSERT(output->get_num_labels()==num);

		for (int32_t i=0; i<num; i++)
			output->set_label(i, classify_example(i));

		return output;
	}

	return NULL;
}
