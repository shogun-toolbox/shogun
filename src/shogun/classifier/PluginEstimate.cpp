/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <lib/common.h>
#include <io/SGIO.h>
#include <features/StringFeatures.h>
#include <labels/Labels.h>
#include <labels/BinaryLabels.h>
#include <labels/RegressionLabels.h>
#include <distributions/LinearHMM.h>
#include <classifier/PluginEstimate.h>

using namespace shogun;

CPluginEstimate::CPluginEstimate(float64_t pos_pseudo, float64_t neg_pseudo)
: CMachine(), m_pos_pseudo(1e-10), m_neg_pseudo(1e-10),
	pos_model(NULL), neg_model(NULL), features(NULL)
{
	m_parameters->add(&m_pos_pseudo,
			"pos_pseudo","pseudo count for positive class");
	m_parameters->add(&m_neg_pseudo,
			"neg_pseudo", "pseudo count for negative class");

	m_parameters->add((CSGObject**) &pos_model,
			"pos_model", "LinearHMM modelling positive class.");
	m_parameters->add((CSGObject**) &neg_model,
			"neg_model", "LinearHMM modelling negative class.");

	m_parameters->add((CSGObject**) &features,
			"features", "String Features.");
}

CPluginEstimate::~CPluginEstimate()
{
	SG_UNREF(pos_model);
	SG_UNREF(neg_model);

	SG_UNREF(features);
}

bool CPluginEstimate::train_machine(CFeatures* data)
{
	ASSERT(m_labels)
	ASSERT(m_labels->get_label_type() == LT_BINARY)
	if (data)
	{
		if (data->get_feature_class() != C_STRING ||
				data->get_feature_type() != F_WORD)
		{
			SG_ERROR("Features not of class string type word\n")
		}

		set_features((CStringFeatures<uint16_t>*) data);
	}
	ASSERT(features)

	SG_UNREF(pos_model);
	SG_UNREF(neg_model);

	pos_model=new CLinearHMM(features);
	neg_model=new CLinearHMM(features);

	SG_REF(pos_model);
	SG_REF(neg_model);

	int32_t* pos_indizes=SG_MALLOC(int32_t, ((CStringFeatures<uint16_t>*) features)->get_num_vectors());
	int32_t* neg_indizes=SG_MALLOC(int32_t, ((CStringFeatures<uint16_t>*) features)->get_num_vectors());

	ASSERT(m_labels->get_num_labels()==features->get_num_vectors())

	int32_t pos_idx=0;
	int32_t neg_idx=0;

	for (int32_t i=0; i<m_labels->get_num_labels(); i++)
	{
		if (((CBinaryLabels*) m_labels)->get_label(i) > 0)
			pos_indizes[pos_idx++]=i;
		else
			neg_indizes[neg_idx++]=i;
	}

	SG_INFO("training using pseudos %f and %f\n", m_pos_pseudo, m_neg_pseudo)
	pos_model->train(pos_indizes, pos_idx, m_pos_pseudo);
	neg_model->train(neg_indizes, neg_idx, m_neg_pseudo);

	SG_FREE(pos_indizes);
	SG_FREE(neg_indizes);

	return true;
}

CBinaryLabels* CPluginEstimate::apply_binary(CFeatures* data)
{
	if (data)
	{
		if (data->get_feature_class() != C_STRING ||
			data->get_feature_type() != F_WORD)
		{
			SG_ERROR("Features not of class string type word\n")
		}

		set_features((CStringFeatures<uint16_t>*) data);
	}

	ASSERT(features)
	SGVector<float64_t> result(features->get_num_vectors());

	for (int32_t vec=0; vec<features->get_num_vectors(); vec++)
		result[vec] = apply_one(vec);

	return new CBinaryLabels(result);
}

float64_t CPluginEstimate::apply_one(int32_t vec_idx)
{
	ASSERT(features)

	int32_t len;
	bool free_vec;
	uint16_t* vector=features->get_feature_vector(vec_idx, len, free_vec);

	if ((!pos_model) || (!neg_model))
		SG_ERROR("model(s) not assigned\n")

	float64_t result=pos_model->get_log_likelihood_example(vector, len) - neg_model->get_log_likelihood_example(vector, len);
	features->free_feature_vector(vector, vec_idx, free_vec);
	return result;
}
