/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/LinearClassifier.h"

CLinearClassifier::CLinearClassifier() : CClassifier(), w(NULL), bias(0), features(NULL)
{
}

CLinearClassifier::~CLinearClassifier()
{
}

CLabels* CLinearClassifier::classify(CLabels* output)
{
	if (features)
	{
		INT num=features->get_num_vectors();
		ASSERT(num>0);

		if (!output)
			output=new CLabels(num);

		ASSERT(output);
		for (INT i=0; i<num; i++)
			output->set_label(i, classify_example(i));

		return output;
	}

	return NULL;
}


bool CLinearClassifier::load(FILE* srcfile)
{
	return false;
}

bool CLinearClassifier::save(FILE* dstfile)
{
	return false;
}
