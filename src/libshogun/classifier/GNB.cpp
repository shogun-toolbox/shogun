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

CGNB::CGNB() :
CClassifier(), m_num_classes(0), m_means(NULL), m_std_devs(NULL), m_label_prob(NULL),
m_labels(NULL),  m_features(NULL), m_rates(NULL), m_feat_vec(NULL)
{

};

CGNB::CGNB(CFeatures* train_examples, CLabels* train_labels) :
CClassifier(), m_num_classes(0), m_means(NULL), m_std_devs(NULL), m_label_prob(NULL),
m_rates(NULL), m_feat_vec(NULL), m_labels(NULL), m_features(NULL)
{
	ASSERT(train_examples->get_num_vectors() == train_labels->get_num_labels());
	set_labels(train_labels);
	set_features((CDotFeatures*)train_examples);
};

CGNB::~CGNB()
{
	//SG_UNREF(labels);
	delete[] m_means;
	delete[] m_rates;
	delete[] m_feat_vec;
	delete[] m_std_devs;
	delete[] m_label_prob;
};

bool CGNB::train(CFeatures* data)
{
	if (data) set_features((CDotFeatures*) data);

	ASSERT(labels);
	m_labels = labels->get_int_labels(m_num_train_labels);
	ASSERT(m_features);

	int32_t min_label = m_labels[0];
	int32_t max_label = m_labels[0];
	int i,j;

	for (i=1; i<m_num_train_labels; i++)
	{
		min_label = CMath::min(min_label, m_labels[i]);
		max_label = CMath::max(max_label, m_labels[i]);
	}

	for (i=1; i<m_num_train_labels; i++)
		m_labels[i]-= min_label;

	m_num_classes = max_label-min_label+1;
	m_min_label = min_label;

	m_dim = m_features->get_dim_feature_space();

	SG_PRINT("num classes: %d\n", m_num_classes);
	SG_PRINT("dim %d\n",m_dim);
	m_means = new float64_t[m_num_classes*m_dim];
	m_std_devs = new float64_t[m_num_classes*m_dim];
	m_rates = new float64_t[m_num_classes];
	m_feat_vec = new float64_t[m_dim];
	m_label_prob = new float64_t[m_num_classes];
	ASSERT(m_means);
	ASSERT(m_std_devs);
	ASSERT(m_rates);

	for (i=0; i<m_num_train_labels; i++)
	{
		m_features->get_feature_vector(&m_feat_vec, &m_dim, i);
		for (j=0; j<m_dim; j++)
			m_means[m_dim*m_labels[i]+j]+=m_feat_vec[j];
		m_label_prob[m_labels[i]]+=1.0;
	}
	for (i=0; i<m_num_classes; i++)
	{
		SG_PRINT("m_label_prob[%d]=%f\n", i,m_label_prob[i]);
		for (j=0; j<m_dim; j++)
		{
			m_means[m_dim*i+j] /= m_label_prob[i];
			SG_PRINT("%d-th label mean of %d-th feature: %f\n", i,j, m_means[i*m_dim+j]);
		}
	}
	for (i=0; i<m_num_train_labels; i++)
	{
		m_features->get_feature_vector(&m_feat_vec, &m_dim, i);
		for (j=0; j<m_dim; j++)
			m_std_devs[m_dim*m_labels[i]+j]+=CMath::pow(m_feat_vec[j]-m_means[m_dim*m_labels[i]+j],2);
	}
	for (i=0; i<m_num_classes; i++)
	{
		for (j=0; j<m_dim; j++)
		{
			m_std_devs[m_dim*i+j] /= m_label_prob[i] > 1 ? m_label_prob[i]-1 : 1;
			SG_PRINT("%d-th label sigma of %d-th feature: %f\n", i,j, m_std_devs[i*m_dim+j]);
		}
	}


	return true;
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
	m_features->get_feature_vector(&m_feat_vec,&m_dim,idx);
	int i,k;
	for(i=0; i<m_num_classes; i++)
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

	for(i=0; i<m_num_classes; i++)
		if (m_rates[i]>m_rates[max_label_idx])
			max_label_idx = i;

	return max_label_idx+m_min_label;
};
