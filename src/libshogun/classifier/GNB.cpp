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
	set_features((CDotFeatures*)train_examples);
};

CGNB::~CGNB()
{
	SG_UNREF(labels);
	delete[] m_means;
	delete[] m_rates;
	delete[] m_feat_vec;
	delete[] m_std_devs;
};

bool CGNB::train(CFeatures* data)
{
	ASSERT(data->get_num_vectors() == num_train_labels);
	m_dim = m_features->get_dim_feature_space();
	m_means = new float64_t[num_classes*m_dim];
	m_std_devs = new float64_t[num_classes*m_dim];
	m_rates = new float64_t[num_classes];
	m_feat_vec = new float64_t[m_dim];
	return false;
}

CLabels* CGNB::classify()
{
	int32_t n = m_features->get_num_vectors();

	// init result labels
	CLabels* result = new CLabels(n);

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
	m_features->get_feature_vector(&m_feat_vec,NULL,idx);
	int i,k;
	for(i=0; i<num_classes; i++)
	{
		if (m_label_prob[i]==0.0)
		{
			m_rates[i] = 0.0;
			continue;
		}
		else m_rates[i] = m_label_prob[i];

		// product all conditional gaussian probabilities
		for(k=0; k<m_dim; k++)
			m_rates[i]*= normal_exp(m_feat_vec[k],i,k)/m_std_devs[i*m_dim+k];
	}

	// find label with maximum rate
	int32_t max_label_idx = 0;

	for(i=0; i<num_classes; i++)
		if (m_rates[i]>m_rates[max_label_idx])
			max_label_idx = i;

	return max_label_idx+min_label;
};
