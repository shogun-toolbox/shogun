/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Hidekazu Oiwa
 */

#include <shogun/classifier/AveragedPerceptron.h>
#include <shogun/features/Labels.h>
#include <shogun/lib/Mathematics.h>

using namespace shogun;

CAveragedPerceptron::CAveragedPerceptron()
: CLinearMachine(), learn_rate(0.1), max_iter(1000)
{
}

CAveragedPerceptron::CAveragedPerceptron(CDotFeatures* traindat, CLabels* trainlab)
: CLinearMachine(), learn_rate(.1), max_iter(1000)
{
	set_features(traindat);
	set_labels(trainlab);
}

CAveragedPerceptron::~CAveragedPerceptron()
{
}

bool CAveragedPerceptron::train(CFeatures* data)
{
	ASSERT(labels);
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n");
		set_features((CDotFeatures*) data);
	}
	ASSERT(features);
	bool converged=false;
	int32_t iter=0;
	int32_t num_train_labels=0;
	int32_t* train_labels=labels->get_int_labels(num_train_labels);
	int32_t num_feat=features->get_dim_feature_space();
	int32_t num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);
	delete[] w;
	w_dim=num_feat;
	w=new float64_t[num_feat];
	float64_t* tmp_w=new float64_t[num_feat];

	float64_t* output=new float64_t[num_vec];
	//start with uniform w, bias=0, tmp_bias=0
	bias=0;
	float64_t tmp_bias=0;
	for (int32_t i=0; i<num_feat; i++)
		w[i]=1.0/num_feat;

	//loop till we either get everything classified right or reach max_iter

	while (!converged && iter<max_iter)
	{
		converged=true;
		for (int32_t i=0; i<num_vec; i++)
		{
			output[i]=apply(i);

			if (CMath::sign<float64_t>(output[i]) != train_labels[i])
			{
				converged=false;
				bias+=learn_rate*train_labels[i];
				features->add_to_dense_vec(learn_rate*train_labels[i], i, w, w_dim);
			}

			// Add current w to tmp_w, and current bias to tmp_bias
			// To calculate the sum of each iteration's w, bias
			for (int32_t j=0; j<num_feat; j++)
				tmp_w[j]+=w[j];
			tmp_bias+=bias;
		}
		iter++;
	}

	if (converged)
		SG_INFO("Averaged Perceptron algorithm converged after %d iterations.\n", iter);
	else
		SG_WARNING("Averaged Perceptron algorithm did not converge after %d iterations.\n", max_iter);

	// calculate and set the average paramter of w, bias
	for (int32_t i=0; i<num_feat; i++)
		w[i]=tmp_w[i]/(num_vec*iter);
	bias=tmp_bias/(num_vec*iter);

	delete[] output;
	delete[] train_labels;
	delete[] tmp_w;

	return converged;
}
