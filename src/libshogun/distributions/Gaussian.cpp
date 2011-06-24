/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Alesis Novik
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#include "lib/config.h"

#ifdef HAVE_LAPACK

#include "distributions/Gaussian.h"
#include "lib/Mathematics.h"
#include "base/Parameter.h"

using namespace shogun;

CGaussian::CGaussian() : CDistribution(), m_constant(0),
m_cov(NULL), m_cov_rows(0), m_cov_cols(0), m_cov_inverse(NULL),
m_cov_inverse_rows(0), m_cov_inverse_cols(0), m_mean(NULL),
m_mean_length(0)
{
}

CGaussian::CGaussian(SGVector<float64_t> mean_vector, SGMatrix<float64_t> cov_matrix) : CDistribution(),
					m_cov_inverse(NULL)
{
	ASSERT(mean_vector.vlen == cov_matrix.num_rows);
	ASSERT(cov_matrix.num_rows == cov_matrix.num_cols);
	m_mean = mean_vector.vector;
	m_cov = cov_matrix.matrix;
	m_mean_length = mean_vector.vlen;
	m_cov_rows = cov_matrix.num_rows;
	m_cov_cols = cov_matrix.num_cols;
	init();
	register_params();
}

void CGaussian::init()
{
	delete[] m_cov_inverse;

	m_cov_inverse_rows = m_cov_cols;
	m_cov_inverse_cols = m_cov_rows;

	m_cov_inverse = new float64_t[m_cov_rows*m_cov_cols];
	memcpy(m_cov_inverse, m_cov, sizeof(float64_t)*m_cov_rows*m_cov_cols);
	int32_t result = clapack_dpotrf(CblasRowMajor, CblasLower, m_cov_rows, m_cov_inverse, m_cov_rows);
	m_constant = 1;

	for (int i = 0; i < m_cov_rows; i++)
		m_constant *= m_cov_inverse[i*m_cov_rows+i];

	m_constant = -CMath::log(m_constant);
	m_constant -= (m_cov_rows/2.0)*CMath::log(2*M_PI);

	result = clapack_dpotri(CblasRowMajor, CblasLower, m_cov_rows, m_cov_inverse, m_cov_rows);
}

CGaussian::~CGaussian()
{
	delete[] m_cov_inverse;
	delete[] m_cov;
	delete[] m_mean;
}

bool CGaussian::train(CFeatures* data)
{
	// init features with data if necessary and assure type is correct
	if (data)
	{
		if (!data->has_property(FP_DOT))
				SG_ERROR("Specified features are not of type CDotFeatures\n");		
		set_features(data);
	}
	CDotFeatures* dotdata = (CDotFeatures *) data;

	delete[] m_mean;
	delete[] m_cov;

	dotdata->get_mean(&m_mean, &m_mean_length);
	dotdata->get_cov(&m_cov, &m_cov_rows, &m_cov_cols);

	init();

	return true;
}

int32_t CGaussian::get_num_model_parameters()
{
	return m_cov_rows*m_cov_cols+m_mean_length;
}

float64_t CGaussian::get_log_model_parameter(int32_t num_param)
{
	if (num_param<m_mean_length)
		return CMath::log(m_mean[num_param]);
	else
		return CMath::log(m_cov[num_param-m_mean_length]);
}

float64_t CGaussian::get_log_derivative(int32_t num_param, int32_t num_example)
{
	SG_NOTIMPLEMENTED;
	return 0;
}

float64_t CGaussian::get_log_likelihood_example(int32_t num_example)
{
	ASSERT(features->has_property(FP_DOT));
	SGVector<float64_t> v=((CDotFeatures *)features)->get_feature_vector(num_example);
	float64_t answer = compute_log_PDF(v.vector, v.vlen);
	v.free_vector();
	return answer;
}

float64_t CGaussian::compute_log_PDF(float64_t* point, int32_t point_len)
{
	ASSERT(m_mean && m_cov);
	ASSERT(point_len == m_mean_length);
	float64_t* difference = new float64_t[m_mean_length];
	memcpy(difference, point, sizeof(float64_t)*m_mean_length);
	float64_t* result = new float64_t[m_mean_length];

	for (int i = 0; i < m_mean_length; i++)
		difference[i] -= m_mean[i];

	cblas_dsymv(CblasRowMajor, CblasLower, m_mean_length, -1.0/2.0, m_cov_inverse, m_mean_length,
				difference, 1, 0, result, 1);

	float64_t answer = m_constant+cblas_ddot(m_mean_length, difference, 1, result, 1);

	delete[] difference;
	delete[] result;

	return answer;
}

void CGaussian::register_params()
{
	m_parameters->add_matrix(&m_cov, &m_cov_rows, &m_cov_cols, "m_cov", "Covariance.");
	m_parameters->add_matrix(&m_cov_inverse, &m_cov_inverse_rows, &m_cov_inverse_cols, "m_cov_inverse", "Covariance inverse.");
	m_parameters->add_vector(&m_mean, &m_mean_length, "m_mean", "Mean.");
	m_parameters->add(&m_constant, "m_constant", "Constant part.");
}
#endif
