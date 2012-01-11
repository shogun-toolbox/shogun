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
#include <shogun/mathematics/Math.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

CGaussianNaiveBayes::CGaussianNaiveBayes() :
CMachine(), m_features(NULL), m_min_label(0),
m_num_classes(0), m_dim(0), m_means(),
m_variances(), m_label_prob(), m_rates()
{

};

CGaussianNaiveBayes::CGaussianNaiveBayes(CFeatures* train_examples, CLabels* train_labels) :
CMachine(), m_features(NULL), m_min_label(0),
m_num_classes(0), m_dim(0), m_means(),
m_variances(), m_label_prob(), m_rates()
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

	m_means.destroy_vector();
	m_rates.destroy_vector();
	m_variances.destroy_vector();
	m_label_prob.destroy_vector();
};

CFeatures* CGaussianNaiveBayes::get_features()
{
	SG_REF(m_features);
	return m_features;
}

void CGaussianNaiveBayes::set_features(CFeatures* features)
{
	if (!features->has_property(FP_DOT))
		SG_ERROR("Specified features are not of type CDotFeatures\n");

	SG_UNREF(m_features);
	SG_REF(features);
	m_features = (CDotFeatures*)features;
}

bool CGaussianNaiveBayes::train(CFeatures* data)
{
	// init features with data if necessary and assure type is correct
	if (data)
	{
		if (!data->has_property(FP_DOT))
				SG_ERROR("Specified features are not of type CDotFeatures\n");
		set_features((CDotFeatures*) data);
	}
	// get int labels to train_labels and check length equality
	ASSERT(labels);
	SGVector<int32_t> train_labels = labels->get_int_labels();
	ASSERT(m_features->get_num_vectors()==train_labels.vlen);

	// init min_label, max_label and loop variables
	int32_t min_label = train_labels.vector[0];
	int32_t max_label = train_labels.vector[0];
	int i,j;

	// find minimal and maximal label
	for (i=1; i<train_labels.vlen; i++)
	{
		min_label = CMath::min(min_label, train_labels.vector[i]);
		max_label = CMath::max(max_label, train_labels.vector[i]);
	}

	// subtract minimal label from all labels
	for (i=0; i<train_labels.vlen; i++)
		train_labels.vector[i]-= min_label;

	// get number of classes, minimal label and dimensionality
	m_num_classes = max_label-min_label+1;
	m_min_label = min_label;
	m_dim = m_features->get_dim_feature_space();

	// allocate memory for distributions' parameters and a priori probability
	m_means.vector = SG_MALLOC(float64_t, m_num_classes*m_dim);
	m_means.vlen = m_num_classes*m_dim;

	m_variances.vector = SG_MALLOC(float64_t, m_num_classes*m_dim);
	m_variances.vlen = m_num_classes*m_dim;

	m_label_prob.vector = SG_MALLOC(float64_t, m_num_classes);
	m_label_prob.vlen = m_num_classes;

	// allocate memory for label rates
	m_rates.vector = SG_MALLOC(float64_t, m_num_classes);
	m_rates.vlen = m_num_classes;

	// assure that memory is allocated
	ASSERT(m_means.vector);
	ASSERT(m_variances.vector);
	ASSERT(m_rates.vector);
	ASSERT(m_label_prob.vector);

	// make arrays filled by zeros before using
	for (i=0;i<m_num_classes*m_dim;i++)
	{
		m_means.vector[i] = 0.0;
		m_variances.vector[i] = 0.0;
	}
	for (i=0;i<m_num_classes;i++)
	{
		m_label_prob.vector[i] = 0.0;
		m_rates.vector[i] = 0.0;
	}

	SGMatrix<float64_t> feature_matrix = m_features->get_computed_dot_feature_matrix();

	// get sum of features among labels
	for (i=0; i<train_labels.vlen; i++)
	{
		for (j=0; j<m_dim; j++)
			m_means.vector[m_dim*train_labels.vector[i]+j]+=feature_matrix.matrix[i*m_dim+j];

		m_label_prob.vector[train_labels.vector[i]]+=1.0;
	}

	// get means of features of labels
	for (i=0; i<m_num_classes; i++)
	{
		for (j=0; j<m_dim; j++)
			m_means.vector[m_dim*i+j] /= m_label_prob.vector[i];
	}

	// compute squared residuals with means available
	for (i=0; i<train_labels.vlen; i++)
	{
		for (j=0; j<m_dim; j++)
			m_variances.vector[m_dim*train_labels.vector[i]+j]+=
					CMath::sq(feature_matrix.matrix[i*m_dim+j]-m_means.vector[m_dim*train_labels.vector[i]+j]);
	}

	// get variance of features of labels
	for (i=0; i<m_num_classes; i++)
	{
		for (j=0; j<m_dim; j++)
			m_variances.vector[m_dim*i+j] /= m_label_prob.vector[i] > 1 ? m_label_prob.vector[i]-1 : 1;
	}

	// get a priori probabilities of labels
	for (i=0; i<m_num_classes; i++)
	{
		m_label_prob.vector[i]/= m_num_classes;
	}

	feature_matrix.free_matrix();
	train_labels.free_vector();

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

	// set features to classify
	set_features(data);

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
		if (m_label_prob.vector[i]==0.0)
		{
			m_rates.vector[i] = 0.0;
			continue;
		}
		else
			m_rates.vector[i] = CMath::log(m_label_prob.vector[i]);

		// product all conditional gaussian probabilities
		for (k=0; k<m_dim; k++)
			m_rates.vector[i]+= CMath::log(normal_exp(feature_vector.vector[k],i,k)/CMath::sqrt(m_variances.vector[i*m_dim+k]));
	}

	// find label with maximum rate
	int32_t max_label_idx = 0;

	for (i=0; i<m_num_classes; i++)
	{
		if (m_rates.vector[i]>m_rates.vector[max_label_idx])
			max_label_idx = i;
	}
	feature_vector.free_vector();

	return max_label_idx+m_min_label;
};
