/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/Perceptron.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"

CPerceptron::CPerceptron()
: CLinearClassifier(), learn_rate(0.1), max_iter(1000)
{
}

CPerceptron::CPerceptron(CRealFeatures* traindat, CLabels* trainlab)
: CLinearClassifier(), learn_rate(.1), max_iter(1000)
{
	set_features(traindat);
	set_labels(trainlab);
}

CPerceptron::~CPerceptron()
{
}

bool CPerceptron::train()
{
	ASSERT(labels);
	ASSERT(features);
	bool converged=false;
	INT iter=0;
	INT num_train_labels=0;
	INT* train_labels=labels->get_int_labels(num_train_labels);
	INT num_feat=features->get_num_features();
	INT num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);
	delete[] w;
	w_dim=num_feat;
	w=new DREAL[num_feat];
	DREAL* output=new DREAL[num_vec];

	//start with uniform w, bias=0
	bias=0;
	for (INT i=0; i<num_feat; i++)
		w[i]=1.0/num_feat;

	//loop till we either get everything classified right or reach max_iter

	while (!converged && iter<max_iter)
	{
		converged=true;
		for (INT i=0; i<num_vec; i++)
			output[i]=classify_example(i);

		for (INT i=0; i<num_vec; i++)
		{
			if (CMath::sign<DREAL>(output[i]) != train_labels[i])
			{
				converged=false;
				INT vlen;
				bool vfree;
				double* vec=features->get_feature_vector(i, vlen, vfree);

				bias+=learn_rate*train_labels[i];
				for (INT j=0; j<num_feat; j++)
					w[j]+=  learn_rate*train_labels[i]*vec[j];

				features->free_feature_vector(vec, i, vfree);
			}
		}

		iter++;
	}

	if (converged)
		SG_INFO("Perceptron algorithm converged after %d iterations.\n", iter);
	else
		SG_WARNING("Perceptron algorithm did not converge after %d iterations.\n", max_iter);

	delete[] output;
	delete[] train_labels;

	return converged;
}
