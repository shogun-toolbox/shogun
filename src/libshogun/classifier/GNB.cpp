/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "GNB.h"
#include "Classifier.h"
#include "features/Features.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"
#include "lib/Signal.h"

using namespace shogun;

CGNB::CGNB() : CClassifier()
{

};

CGNB::CGNB(CFeatures* train_examples, CLabels* train_labels) : CClassifier()
{
	ASSERT(train_examples->get_num_vectors() == train_labels->get_num_labels());
	set_labels(train_labels);
	set_features(train_examples);
};

CGNB::~CGNB()
{
	SG_UNREF(labels);
};

bool CGNB::train(CFeatures* data)
{
	ASSERT(data->get_num_vectors() == num_train_labels);
	m_dim = features->get_dim_feature_space();
	m_means = new float64_t[num_classes][m_dim];
	m_std_devs = new float64_t[num_classes][m_dim];
	m_rates = new float64_t[num_classes];
	m_feat_vec = new float64_t[m_dim];
	return false;
}

CLabels* CGNB::classify()
{
	int32_t n = data->get_num_vectors();

	// init result labels
	CLabels* result = new CLabels(data->get_num_vectors());

	// classify each example of data
	for(int i=0; i<n; i++)
	{
		result->set_label(i,classify_example(i));
	}
	return result;
};

CLabels* CGNB::classify(CFeatures* data)
{
	if (!data)
		SG_ERROR("No features specified\n");
	if (!data->has_property(FP_DOT))
		SG_ERROR("Specified features are not of type CDotFeatures\n");
	set_features((CDotFeatures*)data);
	return classify();
};

float64_t CGNB::classify_example(int32_t idx)
{
	m_features->get_feature_vector(m_feat_vec,m_dim,idx);
	for(j=0; j<num_classes; j++)
	{
		if (m_label_prob[j]==0.0)
		{
			rates[j] = 0.0;
			continue;
		}
		else rates[j] = m_label_prob[j];

		for(k=0; k<m_dim; k++)
			rates[j]*= CMath::exp()/m_std_devs[j];
	}

	// find label with maximum rate
	for(j=0; j<num_classes; j++)
	{

	}
	return max_label_idx+min_label;
};
