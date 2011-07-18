/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/classifier/GaussianNaiveBayes.h>
#include <shogun/machine/Machine.h>
#include <shogun/features/Features.h>
#include <shogun/features/Labels.h>
#include <shogun/lib/Mathematics.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

CGaussianNaiveBayes::CGaussianNaiveBayes() :
CMachine(), m_features(NULL), m_min_label(0), m_labels(NULL),
m_num_train_labels(0), m_num_classes(0), m_dim(0), m_means(NULL),
m_variances(NULL), m_label_prob(NULL), m_rates(NULL)
{

};

CGaussianNaiveBayes::CGaussianNaiveBayes(CFeatures* train_examples, CLabels* train_labels) :
CMachine(), m_features(NULL), m_min_label(0), m_labels(NULL),
m_num_train_labels(0), m_num_classes(0), m_dim(0), m_means(NULL),
m_variances(NULL), m_label_prob(NULL), m_rates(NULL)
{
	ASSERT(train_examples->get_num_vectors() == train_labels->get_num_labels());
	set_labels(train_labels);
	if (!train_examples->has_property(FP_DOT))
		SG_ERROR("Specified features are not of type CDotFeatures\n");
	set_features((CDotFeatures*)train_examples);
};

CGaussianNaiveBayes::~CGaussianNaiveBayes()
{
	SG_UNREF(m_features);
	delete[] m_means;
	delete[] m_rates;
	delete[] m_variances;
	delete[] m_label_prob;
	delete[] m_labels;
};

bool CGaussianNaiveBayes::train(CFeatures* data)
{
	// init features with data if necessary and assure type is correct
	if (data)
	{
		if (!data->has_property(FP_DOT))
				SG_ERROR("Specified features are not of type CDotFeatures\n");
		set_features((CDotFeatures*) data);
	}
	// get int labels to m_labels and check length equality
	ASSERT(labels);
	m_labels = labels->get_int_labels(m_num_train_labels);
	ASSERT(m_features->get_num_vectors()==m_num_train_labels);

	// init min_label, max_label and loop variables
	int32_t min_label = m_labels[0];
	int32_t max_label = m_labels[0];
	int i,j;

	// find minimal and maximal label
	for (i=1; i<m_num_train_labels; i++)
	{
		min_label = CMath::min(min_label, m_labels[i]);
		max_label = CMath::max(max_label, m_labels[i]);
	}

	// subtract minimal label from all labels
	for (i=0; i<m_num_train_labels; i++)
		m_labels[i]-= min_label;

	// get number of classes, minimal label and dimensionality
	m_num_classes = max_label-min_label+1;
	m_min_label = min_label;
	m_dim = m_features->get_dim_feature_space();

	// allocate memory for distributions' parameters and a priori probability
	m_means = new float64_t[m_num_classes*m_dim];
	m_variances = new float64_t[m_num_classes*m_dim];
	m_label_prob = new float64_t[m_num_classes];

	// allocate memory for label rates
	m_rates = new float64_t[m_num_classes];

	// assure that memory is allocated
	ASSERT(m_means);
	ASSERT(m_variances);
	ASSERT(m_rates);
	ASSERT(m_label_prob);

	// make arrays filled by zeros before using
	for (i=0;i<m_num_classes*m_dim;i++)
	{
		m_means[i] = 0.0;
		m_variances[i] = 0.0;
	}
	for (i=0;i<m_num_classes;i++)
	{
		m_label_prob[i] = 0.0;
		m_rates[i] = 0.0;
	}

	SGMatrix<float64_t> feature_matrix = m_features->get_computed_dot_feature_matrix();

	// get sum of features among labels
	for (i=0; i<m_num_train_labels; i++)
	{
		for (j=0; j<m_dim; j++)
			m_means[m_dim*m_labels[i]+j]+=feature_matrix.matrix[i*m_dim+j];

		m_label_prob[m_labels[i]]+=1.0;
	}

	// get means of features of labels
	for (i=0; i<m_num_classes; i++)
	{
		for (j=0; j<m_dim; j++)
			m_means[m_dim*i+j] /= m_label_prob[i];
	}

	// compute squared residuals with means available
	for (i=0; i<m_num_train_labels; i++)
	{
		for (j=0; j<m_dim; j++)
			m_variances[m_dim*m_labels[i]+j]+=
					CMath::sq(feature_matrix.matrix[i*m_dim+j]-m_means[m_dim*m_labels[i]+j]);
	}

	// get variance of features of labels
	for (i=0; i<m_num_classes; i++)
	{
		for (j=0; j<m_dim; j++)
			m_variances[m_dim*i+j] /= m_label_prob[i] > 1 ? m_label_prob[i]-1 : 1;
	}

	// get a priori probabilities of labels
	for (i=0; i<m_num_classes; i++)
	{
		m_label_prob[i]/= m_num_classes;
	}

	return true;
}

CLabels* CGaussianNaiveBayes::apply()
{
	// init number of vectors
	int32_t n = m_features->get_num_vectors();

	// init result labels
	CLabels* result = new CLabels(n);

	// classify each example of data
	for (int i=0; i<n; i++)
		result->set_label(i,apply(i));

	return result;
};

CLabels* CGaussianNaiveBayes::apply(CFeatures* data)
{
	// check data correctness
	if (!data)
		SG_ERROR("No features specified\n");
	if (!data->has_property(FP_DOT))
		SG_ERROR("Specified features are not of type CDotFeatures\n");

	// set features to classify
	set_features((CDotFeatures*)data);

	// classify using features
	return apply();
};

float64_t CGaussianNaiveBayes::apply(int32_t idx)
{
	// get [idx] feature vector
	SGVector<float64_t> feature_vector = m_features->get_computed_dot_feature_vector(idx);

	// init loop variables
	int i,k;

	// rate all labels
	for (i=0; i<m_num_classes; i++)
	{
		// set rate to 0.0 if a priori probability is 0.0 and continue
		if (m_label_prob[i]==0.0)
		{
			m_rates[i] = 0.0;
			continue;
		}
		else
			m_rates[i] = m_label_prob[i];

		// product all conditional gaussian probabilities
		for (k=0; k<m_dim; k++)
			m_rates[i]*= normal_exp(feature_vector.vector[k],i,k)/CMath::sqrt(m_variances[i*m_dim+k]);
	}

	// find label with maximum rate
	int32_t max_label_idx = 0;

	for (i=0; i<m_num_classes; i++)
	{
		if (m_rates[i]>m_rates[max_label_idx])
			max_label_idx = i;
	}

	return max_label_idx+m_min_label;
};
