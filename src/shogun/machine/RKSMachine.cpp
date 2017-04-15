/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/machine/RKSMachine.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{

CRKSMachine::CRKSMachine()
{
	init(NULL, NULL, SGMatrix<float64_t>());
}

CRKSMachine::CRKSMachine(CFeatures* dataset, CLabels* labels, int32_t num_samples,
	float64_t (*phi)(SGVector<float64_t>, SGVector<float64_t>),	
	SGVector<float64_t> (*p)())
{
	SGMatrix<float64_t> random_params = generate_random_coefficients(p, num_samples);
	CDenseFeatures<float64_t>* converted_dataset = convert_data(dataset, phi, random_params);
	m_phi = phi;
	init(converted_dataset, labels, random_params);
}

CRKSMachine::CRKSMachine(CFeatures* dataset, CLabels* labels,
	float64_t (*phi)(SGVector<float64_t>, SGVector<float64_t>),	SGMatrix<float64_t> a)	
{
	CDenseFeatures<float64_t>* converted_dataset = convert_data(dataset, phi, a);
	m_phi = phi;
	init(converted_dataset, labels, a);
}

CRKSMachine::~CRKSMachine()
{
	SG_UNREF(m_dataset);
}

void CRKSMachine::init(CDenseFeatures<float64_t>* dataset, CLabels* labels,
	SGMatrix<float64_t> random_params) 
{
	m_dataset = dataset;
	SG_REF(dataset);
	
	set_labels(labels);
	random_coeff = random_params;
	SG_ADD((CSGObject** ) &m_dataset , "dataset", "The transformed dataset",
			MS_NOT_AVAILABLE);	
	m_parameters->add(&random_coeff, "random_coeff", "Random coefficients");
}

void CRKSMachine::set_features(CFeatures* feats)
{
	ASSERT(feats);
	ASSERT(feats->has_property(FP_DOT));

	if (m_phi==NULL)
		SG_ERROR("Phi function is not set. Please specify it using the constructor \
				 or the set_phi_function() method.\n");

	SG_UNREF(m_dataset);
	m_dataset = (CDenseFeatures<float64_t>* ) convert_data(feats, m_phi, random_coeff);
	SG_REF(m_dataset);
}

CDenseFeatures<float64_t>* CRKSMachine::convert_data(CFeatures* feats,
	float64_t (*phi)(SGVector<float64_t>, SGVector<float64_t>),
	SGMatrix<float64_t> random_params)
{
	CDotFeatures* dataset = (CDotFeatures* ) feats;
	int32_t num_samples = random_params.num_cols;
	SGMatrix<float64_t> conversion_mat(num_samples, dataset->get_num_vectors());
	for (index_t i=0; i<dataset->get_num_vectors(); i++)
	{
		SGVector<float64_t> vec = dataset->get_computed_dot_feature_vector(i);
		for (index_t j=0; j<num_samples; j++)
		{
			SGVector<float64_t> params(random_params.get_column_vector(j),
					random_params.num_rows, false);
			conversion_mat(j,i) = phi(vec, params);
		}
	}

	return new CDenseFeatures<float64_t>(conversion_mat);
}

SGMatrix<float64_t> CRKSMachine::generate_random_coefficients(
	SGVector<float64_t> (*p)(), int32_t num_samples)
{
	ASSERT(num_samples>0);
	SGVector<float64_t> vec = p();
	SGMatrix<float64_t> random_params(vec.vlen, num_samples);
	for (index_t dim=0; dim<random_params.num_rows; dim++)
		random_params(dim, 0) = vec[dim];

	for (index_t sample=1; sample<num_samples; sample++)
	{
		vec = p();
		for (index_t dim=0; dim<random_params.num_rows; dim++)
			random_params(dim, sample) = vec[dim];	
	}
	return random_params;
}

const char* CRKSMachine::get_name() const
{
	return "RKSMachine";
}

CDenseFeatures<float64_t>* CRKSMachine::get_features() const
{
	SG_REF(m_dataset);
	return m_dataset;
}
}
