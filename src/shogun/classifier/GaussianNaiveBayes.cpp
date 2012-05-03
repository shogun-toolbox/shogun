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

	m_means.destroy_matrix();
	m_variances.destroy_matrix();
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
	ASSERT(m_labels);
	SGVector<int32_t> train_labels = m_labels->get_int_labels();
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
	m_means.matrix = SG_MALLOC(float64_t, m_num_classes*m_dim);
	m_means.num_rows = m_dim;
	m_means.num_cols = m_num_classes;

	m_variances.matrix = SG_MALLOC(float64_t, m_num_classes*m_dim);
	m_variances.num_rows = m_dim;
	m_variances.num_cols = m_num_classes;

	m_label_prob=SGVector<float64_t>(m_num_classes);

	// allocate memory for label rates
	m_rates=SGVector<float64_t>(m_num_classes);

	// assure that memory is allocated
	ASSERT(m_means.matrix);
	ASSERT(m_variances.matrix);

	// make arrays filled by zeros before using
	m_means.zero();
	m_variances.zero();
	m_label_prob.zero();
	m_rates.zero();

	// number of iterations in all cycles
	int32_t max_progress = 2 * train_labels.vlen + 2 * m_num_classes;
	
	// current progress
	int32_t progress = 0;	
	SG_PROGRESS(progress, 0, max_progress);

	// get sum of features among labels
	for (i=0; i<train_labels.vlen; i++)
	{
		SGVector<float64_t> fea = m_features->get_computed_dot_feature_vector(i);
		for (j=0; j<m_dim; j++)
			m_means(j, train_labels.vector[i]) += fea.vector[j];

		m_label_prob.vector[train_labels.vector[i]]+=1.0;

		progress++;
		SG_PROGRESS(progress, 0, max_progress);
	}

	// get means of features of labels
	for (i=0; i<m_num_classes; i++)
	{
		for (j=0; j<m_dim; j++)
			m_means(j, i) /= m_label_prob.vector[i];

		progress++;
		SG_PROGRESS(progress, 0, max_progress);
	}

	// compute squared residuals with means available
	for (i=0; i<train_labels.vlen; i++)
	{
		SGVector<float64_t> fea = m_features->get_computed_dot_feature_vector(i);
		for (j=0; j<m_dim; j++)
		{
			m_variances(j, train_labels.vector[i]) += 
				CMath::sq(fea[j]-m_means(j, train_labels.vector[i]));
		}

		progress++;
		SG_PROGRESS(progress, 0, max_progress);
	}	

	// get variance of features of labels
	for (i=0; i<m_num_classes; i++)
	{
		for (j=0; j<m_dim; j++)
			m_variances(j, i) /= m_label_prob.vector[i] > 1 ? m_label_prob.vector[i]-1 : 1;
		
		// get a priori probabilities of labels
		m_label_prob.vector[i]/= m_num_classes;

		progress++;
		SG_PROGRESS(progress, 0, max_progress);
	}
	SG_DONE();

	return true;
}

CLabels* CGaussianNaiveBayes::apply()
{
	// init number of vectors
	int32_t num_vectors = m_features->get_num_vectors();

	// init result labels
	CLabels* result = new CLabels(num_vectors);

	// classify each example of data
	SG_PROGRESS(0, 0, num_vectors);
	for (int i = 0; i < num_vectors; i++)
	{
		result->set_label(i,apply(i));
		SG_PROGRESS(i + 1, 0, num_vectors);
	}
	SG_DONE();
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
			m_rates.vector[i]+= CMath::log(normal_exp(feature_vector.vector[k],i,k)/CMath::sqrt(m_variances(k, i)));
	}

	// find label with maximum rate
	int32_t max_label_idx = 0;

	for (i=0; i<m_num_classes; i++)
	{
		if (m_rates.vector[i]>m_rates.vector[max_label_idx])
			max_label_idx = i;
	}

	return max_label_idx+m_min_label;
};
