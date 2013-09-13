/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/classifier/RKSClassifier.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{

const float64_t CRKSClassifier::DEFAULT_EPSILON = 0.01;
const float64_t CRKSClassifier::DEFAULT_C = 0.1;

CRKSClassifier::CRKSClassifier() : CRKSMachine()
{
	init();
}

CRKSClassifier::CRKSClassifier(CFeatures* dataset, CLabels* labels, int32_t num_samples,
	float64_t (*phi)(SGVector<float64_t>, SGVector<float64_t>),	
	SGVector<float64_t> (*p)()) : CRKSMachine(dataset, labels, num_samples, phi, p)
{
	init();
}

CRKSClassifier::CRKSClassifier(CFeatures* dataset, CLabels* labels,
	float64_t (*phi)(SGVector<float64_t>, SGVector<float64_t>),	SGMatrix<float64_t> a)
	: CRKSMachine(dataset, labels, phi, a)
{
	init();
}

void CRKSClassifier::init()
{
	linear_classifier = new CLibLinear(L2R_LR);	
	SG_REF(linear_classifier);

	linear_classifier->set_epsilon(DEFAULT_EPSILON);
	linear_classifier->set_features(m_dataset);
	linear_classifier->set_labels(m_labels);
	linear_classifier->set_C(DEFAULT_C, DEFAULT_C);

	SG_ADD((CSGObject** ) &linear_classifier, "linear_classifier", "A linear classifier",
			MS_NOT_AVAILABLE);
}

CRKSClassifier::~CRKSClassifier()
{
	SG_UNREF(linear_classifier);
}

bool CRKSClassifier::train(CFeatures* feats)
{
	if (feats!=NULL)
	{
		set_features(feats);
		linear_classifier->set_features(m_dataset);
	}
	
	return linear_classifier->train();
}

CLabels* CRKSClassifier::apply(CFeatures* feats)
{
	return apply_binary(feats);
}

CBinaryLabels* CRKSClassifier::apply_binary(CFeatures* data)
{
	if (data!=NULL)
	{
		set_features(data);
		linear_classifier->set_features(m_dataset);
	}
	return linear_classifier->apply_binary();
}

float64_t CRKSClassifier::apply_one(int32_t vec_idx)
{
	return linear_classifier->apply_one(vec_idx);
}

const char* CRKSClassifier::get_name() const
{
	return "RKSClassifier";
}

void CRKSClassifier::set_C(float64_t C)
{
	linear_classifier->set_C(C, C);
}

void CRKSClassifier::set_epsilon(float64_t epsilon)
{
	linear_classifier->set_epsilon(epsilon);
}
}
