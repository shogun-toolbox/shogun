/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Alesis Novik
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK

#include <shogun/distributions/Gaussian.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

CGaussian::CGaussian() : CDistribution(), m_constant(0), m_d(), m_u(), m_mean(), m_cov_type(FULL)
{
	register_params();
}

CGaussian::CGaussian(const SGVector<float64_t> mean, SGMatrix<float64_t> cov,
					ECovType cov_type) : CDistribution(), m_d(), m_u(), m_cov_type(cov_type)
{
	ASSERT(mean.vlen==cov.num_rows)
	ASSERT(cov.num_rows==cov.num_cols)

	m_mean=mean;

	if (cov.num_rows==1)
		m_cov_type=SPHERICAL;

	decompose_cov(cov);
	init();
	register_params();
}

void CGaussian::init()
{
	m_constant=CMath::log(2*M_PI)*m_mean.vlen;
	switch (m_cov_type)
	{
		case FULL:
		case DIAG:
			for (int32_t i=0; i<m_d.vlen; i++)
				m_constant+=CMath::log(m_d.vector[i]);
			break;
		case SPHERICAL:
			m_constant+=m_mean.vlen*CMath::log(m_d.vector[0]);
			break;
	}
}

CGaussian::~CGaussian()
{
}

bool CGaussian::train(CFeatures* data)
{
	// init features with data if necessary and assure type is correct
	if (data)
	{
		if (!data->has_property(FP_DOT))
				SG_ERROR("Specified features are not of type CDotFeatures\n")
		set_features(data);
	}

	CDotFeatures* dotdata=(CDotFeatures *) data;
	m_mean=dotdata->get_mean();
	SGMatrix<float64_t> cov=dotdata->get_cov();
	decompose_cov(cov);
	init();
	return true;
}

int32_t CGaussian::get_num_model_parameters()
{
	switch (m_cov_type)
	{
		case FULL:
			return m_u.num_rows*m_u.num_cols+m_d.vlen+m_mean.vlen;
		case DIAG:
			return m_d.vlen+m_mean.vlen;
		case SPHERICAL:
			return 1+m_mean.vlen;
	}
	return 0;
}

float64_t CGaussian::get_log_model_parameter(int32_t num_param)
{
	SG_NOTIMPLEMENTED
	return 0;
}

float64_t CGaussian::get_log_derivative(int32_t num_param, int32_t num_example)
{
	SG_NOTIMPLEMENTED
	return 0;
}

float64_t CGaussian::get_log_likelihood_example(int32_t num_example)
{
	ASSERT(features->has_property(FP_DOT))
	SGVector<float64_t> v=((CDotFeatures *)features)->get_computed_dot_feature_vector(num_example);
	float64_t answer=compute_log_PDF(v);
	return answer;
}

float64_t CGaussian::compute_log_PDF(SGVector<float64_t> point)
{
	ASSERT(m_mean.vector && m_d.vector)
	ASSERT(point.vlen == m_mean.vlen)
	float64_t* difference=SG_MALLOC(float64_t, m_mean.vlen);
	memcpy(difference, point.vector, sizeof(float64_t)*m_mean.vlen);

	for (int32_t i = 0; i < m_mean.vlen; i++)
		difference[i] -= m_mean.vector[i];

	float64_t answer=m_constant;

	if (m_cov_type==FULL)
	{
		float64_t* temp_holder=SG_MALLOC(float64_t, m_d.vlen);
		cblas_dgemv(CblasRowMajor, CblasNoTrans, m_d.vlen, m_d.vlen,
					1, m_u.matrix, m_d.vlen, difference, 1, 0, temp_holder, 1);

		for (int32_t i=0; i<m_d.vlen; i++)
			answer+=temp_holder[i]*temp_holder[i]/m_d.vector[i];

		SG_FREE(temp_holder);
	}
	else if (m_cov_type==DIAG)
	{
		for (int32_t i=0; i<m_mean.vlen; i++)
			answer+=difference[i]*difference[i]/m_d.vector[i];
	}
	else
	{
		for (int32_t i=0; i<m_mean.vlen; i++)
			answer+=difference[i]*difference[i]/m_d.vector[0];
	}

	SG_FREE(difference);

	return -0.5*answer;
}

SGVector<float64_t> CGaussian::get_mean()
{
	return m_mean;
}

void CGaussian::set_mean(SGVector<float64_t> mean)
{
	if (mean.vlen==1)
		m_cov_type=SPHERICAL;

	m_mean=mean;
}

void CGaussian::set_cov(SGMatrix<float64_t> cov)
{
	ASSERT(cov.num_rows==cov.num_cols)
	ASSERT(cov.num_rows==m_mean.vlen)
	decompose_cov(cov);
	init();
}

void CGaussian::set_d(const SGVector<float64_t> d)
{
	m_d = d;
	init();
}

SGMatrix<float64_t> CGaussian::get_cov()
{
	float64_t* cov=SG_MALLOC(float64_t, m_mean.vlen*m_mean.vlen);
	memset(cov, 0, sizeof(float64_t)*m_mean.vlen*m_mean.vlen);

	if (m_cov_type==FULL)
	{
		if (!m_u.matrix)
			SG_ERROR("Unitary matrix not set\n")

		float64_t* temp_holder=SG_MALLOC(float64_t, m_d.vlen*m_d.vlen);
		float64_t* diag_holder=SG_MALLOC(float64_t, m_d.vlen*m_d.vlen);
		memset(diag_holder, 0, sizeof(float64_t)*m_d.vlen*m_d.vlen);
		for(int32_t i=0; i<m_d.vlen; i++)
			diag_holder[i*m_d.vlen+i]=m_d.vector[i];

		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
					m_d.vlen, m_d.vlen, m_d.vlen, 1, m_u.matrix, m_d.vlen,
					diag_holder, m_d.vlen, 0, temp_holder, m_d.vlen);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m_d.vlen, m_d.vlen, m_d.vlen, 1, temp_holder, m_d.vlen,
					m_u.matrix, m_d.vlen, 0, cov, m_d.vlen);

		SG_FREE(diag_holder);
		SG_FREE(temp_holder);
	}
	else if (m_cov_type==DIAG)
	{
		for (int32_t i=0; i<m_d.vlen; i++)
			cov[i*m_d.vlen+i]=m_d.vector[i];
	}
	else
	{
		for (int32_t i=0; i<m_mean.vlen; i++)
			cov[i*m_mean.vlen+i]=m_d.vector[0];
	}
	return SGMatrix<float64_t>(cov, m_mean.vlen, m_mean.vlen, false);//fix needed
}

void CGaussian::register_params()
{
	m_parameters->add(&m_u, "m_u", "Unitary matrix.");
	m_parameters->add(&m_d, "m_d", "Diagonal.");
	m_parameters->add(&m_mean, "m_mean", "Mean.");
	m_parameters->add(&m_constant, "m_constant", "Constant part.");
	m_parameters->add((machine_int_t*)&m_cov_type, "m_cov_type", "Covariance type.");
}

void CGaussian::decompose_cov(SGMatrix<float64_t> cov)
{
	switch (m_cov_type)
	{
		case FULL:
			m_u=SGMatrix<float64_t>(cov.num_rows,cov.num_rows);
			memcpy(m_u.matrix, cov.matrix, sizeof(float64_t)*cov.num_rows*cov.num_rows);

			m_d.vector=SGMatrix<float64_t>::compute_eigenvectors(m_u.matrix, cov.num_rows, cov.num_rows);
			m_d.vlen=cov.num_rows;
			m_u.num_rows=cov.num_rows;
			m_u.num_cols=cov.num_rows;
			break;
		case DIAG:
			m_d.vector=SG_MALLOC(float64_t, cov.num_rows);

			for (int32_t i=0; i<cov.num_rows; i++)
				m_d.vector[i]=cov.matrix[i*cov.num_rows+i];

			m_d.vlen=cov.num_rows;
			break;
		case SPHERICAL:
			m_d.vector=SG_MALLOC(float64_t, 1);

			m_d.vector[0]=cov.matrix[0];
			m_d.vlen=1;
			break;
	}
}

SGVector<float64_t> CGaussian::sample()
{
	float64_t* r_matrix=SG_MALLOC(float64_t, m_mean.vlen*m_mean.vlen);
	memset(r_matrix, 0, m_mean.vlen*m_mean.vlen*sizeof(float64_t));

	switch (m_cov_type)
	{
		case FULL:
		case DIAG:
			for (int32_t i=0; i<m_mean.vlen; i++)
				r_matrix[i*m_mean.vlen+i]=CMath::sqrt(m_d.vector[i]);

			break;
		case SPHERICAL:
			for (int32_t i=0; i<m_mean.vlen; i++)
				r_matrix[i*m_mean.vlen+i]=CMath::sqrt(m_d.vector[0]);

			break;
	}

	float64_t* random_vec=SG_MALLOC(float64_t, m_mean.vlen);

	for (int32_t i=0; i<m_mean.vlen; i++)
		random_vec[i]=CMath::randn_double();

	if (m_cov_type==FULL)
	{
		float64_t* temp_matrix=SG_MALLOC(float64_t, m_d.vlen*m_d.vlen);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					m_d.vlen, m_d.vlen, m_d.vlen, 1, m_u.matrix, m_d.vlen,
					r_matrix, m_d.vlen, 0, temp_matrix, m_d.vlen);
		SG_FREE(r_matrix);
		r_matrix=temp_matrix;
	}

	float64_t* samp=SG_MALLOC(float64_t, m_mean.vlen);

	cblas_dgemv(CblasRowMajor, CblasNoTrans, m_mean.vlen, m_mean.vlen,
				1, r_matrix, m_mean.vlen, random_vec, 1, 0, samp, 1);

	for (int32_t i=0; i<m_mean.vlen; i++)
		samp[i]+=m_mean.vector[i];

	SG_FREE(random_vec);
	SG_FREE(r_matrix);

	return SGVector<float64_t>(samp, m_mean.vlen, false);//fix needed
}

#endif // HAVE_LAPACK
