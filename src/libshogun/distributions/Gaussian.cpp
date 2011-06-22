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
m_d(NULL), m_d_length(0), m_u(NULL), m_u_rows(0), m_u_cols(0),
m_mean(NULL), m_mean_length(0), m_cov_type(FULL)
{
	register_params();
}

<<<<<<< HEAD
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
=======
CGaussian::CGaussian(float64_t* mean, int32_t mean_length,
					float64_t* cov, int32_t cov_rows, int32_t cov_cols,
					ECovType cov_type) : CDistribution(),
					m_d(NULL), m_d_length(0), m_u(NULL), m_u_rows(0), m_u_cols(0),
					m_cov_type(cov_type)
{
	ASSERT(mean_length == cov_rows);
	ASSERT(cov_rows == cov_cols);
	m_mean=new float64_t[mean_length];
	memcpy(m_mean, mean, sizeof(float64_t)*mean_length);
	m_mean_length=mean_length;

	if (cov_rows==1)
		m_cov_type=SPHERICAL;

	decompose_cov(cov, cov_rows);
>>>>>>> Rewritten Gaussian class to work with different covariance types in log domain.
	init();
	register_params();
}

void CGaussian::init()
{
	m_constant=CMath::log(2*M_PI)*m_mean_length;
	switch (m_cov_type)
	{
		case FULL:
		case DIAG:
			for (int i=0; i<m_d_length; i++)
				m_constant+=CMath::log(m_d[i]);
			break;
		case SPHERICAL:
			m_constant+=m_mean_length*CMath::log(*m_d);
			break;
	}
}

CGaussian::~CGaussian()
{
	delete[] m_d;
	delete[] m_u;
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

	float64_t* cov;
	int32_t cov_rows;
	int32_t cov_cols;

	dotdata->get_mean(&m_mean, &m_mean_length);
	dotdata->get_cov(&cov, &cov_rows, &cov_cols);

	decompose_cov(cov, cov_rows);
	delete[] cov;

	init();

	return true;
}

int32_t CGaussian::get_num_model_parameters()
{
	switch (m_cov_type)
	{
		case FULL:
			return m_u_rows*m_u_cols+m_d_length+m_mean_length;
		case DIAG:
			return m_d_length+m_mean_length;
		case SPHERICAL:
			return 1+m_mean_length;
	}
	return 0;
}

float64_t CGaussian::get_log_model_parameter(int32_t num_param)
{
	SG_NOTIMPLEMENTED;
	return 0;
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
	ASSERT(m_mean && m_d);
	ASSERT(point_len == m_mean_length);
	float64_t* difference=new float64_t[m_mean_length];
	memcpy(difference, point, sizeof(float64_t)*m_mean_length);

	for (int i = 0; i < m_mean_length; i++)
		difference[i] -= m_mean[i];

	float64_t answer=m_constant;

	if (m_cov_type==FULL)
	{
		float64_t* temp_holder=new float64_t[m_d_length];
		cblas_dgemv(CblasRowMajor, CblasNoTrans, m_d_length, m_d_length,
					1, m_u, m_d_length, difference, 1, 0, temp_holder, 1);

		for (int i=0; i<m_d_length; i++)
			answer+=temp_holder[i]*temp_holder[i]/m_d[i];

		delete[] temp_holder;
	}
	else if (m_cov_type==DIAG)
	{
		for (int i=0; i<m_mean_length; i++)
			answer+=difference[i]*difference[i]/m_d[i];
	}
	else
	{
		for (int i=0; i<m_mean_length; i++)
			answer+=difference[i]*difference[i]/(*m_d);
	}

	delete[] difference;

	return -0.5*answer;
}

void CGaussian::get_cov(float64_t** cov, int32_t* cov_rows, int32_t* cov_cols)
{
	*cov=new float64_t[m_mean_length*m_mean_length];
	memset(*cov, 0, sizeof(float64_t)*m_mean_length*m_mean_length);

	if (m_cov_type==FULL)
	{
		if (!m_u)
			SG_ERROR("Unitary matrix not set\n");

		float64_t* temp_holder=new float64_t[m_d_length*m_d_length];
		float64_t* diag_holder=new float64_t[m_d_length*m_d_length];
		memset(diag_holder, 0, sizeof(float64_t)*m_d_length*m_d_length);
		for(int i=0; i<m_d_length; i++)
			diag_holder[i*m_d_length+i]=m_d[i];

		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
					m_d_length, m_d_length, m_d_length, 1, m_u, m_d_length,
					diag_holder, m_d_length, 0, temp_holder, m_d_length);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m_d_length, m_d_length, m_d_length, 1, temp_holder, m_d_length,
					m_u, m_d_length, 0, *cov, m_d_length);

		delete[] diag_holder;
		delete[] temp_holder;
	}
	else if (m_cov_type==DIAG)
	{
		for (int i=0; i<m_d_length; i++)
			(*cov)[i*m_d_length+i]=m_d[i];
	}
	else
	{
		for (int i=0; i<m_mean_length; i++)
			(*cov)[i*m_mean_length+i]=*m_d;
	}

	*cov_rows=m_mean_length;
	*cov_cols=m_mean_length;
}

void CGaussian::register_params()
{
	m_parameters->add_matrix(&m_u, &m_u_rows, &m_u_cols, "m_u", "Unitary matrix.");
	m_parameters->add_vector(&m_d, &m_d_length, "m_d", "Diagonal.");
	m_parameters->add_vector(&m_mean, &m_mean_length, "m_mean", "Mean.");
	m_parameters->add(&m_constant, "m_constant", "Constant part.");
	m_parameters->add((machine_int_t*)&m_cov_type, "m_cov_type", "Covariance type.");
}

void CGaussian::decompose_cov(float64_t* cov, int32_t cov_size)
{
	delete[] m_d;
	switch (m_cov_type)
	{
		case FULL:
			delete[] m_u;
			m_u=new float64_t[cov_size*cov_size];
			memcpy(m_u, cov, sizeof(float64_t)*cov_size*cov_size);

			m_d=CMath::compute_eigenvectors(m_u, cov_size, cov_size);
			m_d_length=cov_size;
			m_u_rows=cov_size;
			m_u_cols=cov_size;
			break;
		case DIAG:
			m_d=new float64_t[cov_size];

			for (int i=0; i<cov_size; i++)
				m_d[i]=cov[i*cov_size+i];
			
			m_d_length=cov_size;
			break;
		case SPHERICAL:
			m_d=new float64_t;

			*m_d=cov[0];
			m_d_length=1;
			break;
	}
}

void CGaussian::sample(float64_t** samp, int32_t* samp_length)
{
	float64_t* r_matrix=new float64_t[m_mean_length*m_mean_length];
	memset(r_matrix, 0, m_mean_length*m_mean_length*sizeof(float64_t));

	switch (m_cov_type)
	{
		case FULL:
		case DIAG:
			for (int i=0; i<m_mean_length; i++)
				r_matrix[i*m_mean_length+i]=CMath::sqrt(m_d[i]);

			break;
		case SPHERICAL:
			for (int i=0; i<m_mean_length; i++)
				r_matrix[i*m_mean_length+i]=CMath::sqrt(*m_d);

			break;
	}

	float64_t* random_vec=new float64_t[m_mean_length];

	for (int i=0; i<m_mean_length; i++)
		random_vec[i]=CMath::randn_double();

	if (m_cov_type==FULL)
	{
		float64_t* temp_matrix=new float64_t[m_d_length*m_d_length];
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m_d_length, m_d_length, m_d_length, 1, m_u, m_d_length,
					r_matrix, m_d_length, 0, temp_matrix, m_d_length);
		delete[] r_matrix;
		r_matrix=temp_matrix;
	}
	
	*samp=new float64_t[m_mean_length];
	*samp_length=m_mean_length;

	cblas_dgemv(CblasRowMajor, CblasNoTrans, m_mean_length, m_mean_length,
				1, r_matrix, m_mean_length, random_vec, 1, 0, *samp, 1);

	for (int i=0; i<m_mean_length; i++)
		*samp[i]+=m_mean[i];

	delete[] random_vec;
	delete[] r_matrix;
}

#endif
