/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/classifier/Perceptron.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/Signal.h>
#include <shogun/base/range.h>

using namespace shogun;

CPerceptron::CPerceptron()
: CLinearMachine(), learn_rate(0.1), max_iter(1000), m_initialize_hyperplane(true)
{
}

CPerceptron::CPerceptron(CDotFeatures* traindat, CLabels* trainlab)
: CLinearMachine(), learn_rate(0.1), max_iter(1000), m_initialize_hyperplane(true)
{
	set_features(traindat);
	set_labels(trainlab);
}

CPerceptron::~CPerceptron()
{
}

bool CPerceptron::train_machine(CFeatures* data)
{
	ASSERT(m_labels)
	ASSERT(m_labels->get_label_type() == LT_BINARY)

	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n")
		set_features((CDotFeatures*) data);
	}

	ASSERT(features)
	bool converged=false;
	index_t iter = 0;
	SGVector<index_t> train_labels =
	    ((CBinaryLabels*)m_labels)->get_int_labels();
	index_t num_feat = features->get_dim_feature_space();
	index_t num_vec = features->get_num_vectors();

	ASSERT(num_vec==train_labels.vlen)
	float64_t* output=SG_MALLOC(float64_t, num_vec);

	SGVector<float64_t> w = get_w();
	if (m_initialize_hyperplane)
	{
		w = SGVector<float64_t>(num_feat);
		//start with uniform w, bias=0
		bias=0;
		for (index_t i = 0; i < num_feat; i++)
			w.vector[i]=1.0/num_feat;
	}


	//loop till we either get everything classified right or reach max_iter
	while (!(cancel_computation()) && (!converged && iter < max_iter))
	{
		converged=true;
		for (auto example_idx : features->index_iterator())
		{
			const auto predicted_label = features->dense_dot(example_idx, w.vector, w.vlen) + bias;
			const auto true_label = train_labels[example_idx];
			output[example_idx] = predicted_label;

			if (CMath::sign<float64_t>(predicted_label) != true_label)
			{
				converged = false;
				const auto gradient = learn_rate * train_labels[example_idx];
				bias += gradient;
				features->add_to_dense_vec(gradient, example_idx, w.vector, w.vlen);
			}
		}

		iter++;
	}

	if (converged)
		SG_INFO("Perceptron algorithm converged after %d iterations.\n", iter)
	else
		SG_WARNING("Perceptron algorithm did not converge after %d iterations.\n", max_iter)

	SG_FREE(output);

	set_w(w);

	return converged;
}

void CPerceptron::set_initialize_hyperplane(bool initialize_hyperplane)
{
	m_initialize_hyperplane = initialize_hyperplane;
}

bool CPerceptron::get_initialize_hyperplane()
{
	return m_initialize_hyperplane;
}
