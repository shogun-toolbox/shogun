/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/SparseLinearClassifier.h"

CSparseLinearClassifier::CSparseLinearClassifier() : CClassifier(), w_dim(0), w(NULL), bias(0), features(NULL)
{
}

CSparseLinearClassifier::~CSparseLinearClassifier()
{
    delete[] w;
}

CLabels* CSparseLinearClassifier::classify(CLabels* output)
{
	if (features)
	{
		INT num=features->get_num_vectors();
		ASSERT(num>0);
		ASSERT(w_dim == features->get_num_features());

		if (!output)
			output=new CLabels(num);

		ASSERT(output && output->get_num_labels() == num);
		for (INT i=0; i<num; i++)
			output->set_label(i, classify_example(i));

		return output;
	}

	return NULL;
}
