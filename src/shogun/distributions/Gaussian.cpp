/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Weijie Lin, Alesis Novik, Heiko Strathmann,
 *          Evgeniy Andreev, Viktor Gal, Evan Shelhamer, Bjoern Esser
 */
#include <shogun/lib/config.h>

#include <shogun/base/Parameter.h>
#include <shogun/distributions/Gaussian.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/mathematics/RandomNamespace.h>

using namespace shogun;
using namespace linalg;

Gaussian::Gaussian() : RandomMixin<Distribution>(), m_constant(0), m_d(), m_u(), m_mean(), m_cov_type(FULL)
{
	register_params();
}

Gaussian::Gaussian(
    const SGVector<float64_t> mean, SGMatrix<float64_t> cov, ECovType cov_type)
    : RandomMixin<Distribution>()
{
	ASSERT(mean.vlen==cov.num_rows)
	ASSERT(cov.num_rows==cov.num_cols)
	m_d=SGVector<float64_t>();
	m_u=SGMatrix<float64_t>();
	m_cov_type=cov_type;

	m_mean=mean;

	if (cov.num_rows==1)
		m_cov_type=SPHERICAL;

	decompose_cov(cov);
	init();
	register_params();
}

void Gaussian::init()
{
	m_constant = std::log(2 * M_PI) * m_mean.vlen;
	switch (m_cov_type)
	{
		case FULL:
		case DIAG:
			for (const auto& v: m_d)
			    m_constant += std::log(v);
		    break;
	    case SPHERICAL:
		    m_constant += m_mean.vlen * std::log(m_d.vector[0]);
		    break;
	}
}

Gaussian::~Gaussian()
{
}

bool Gaussian::train(std::shared_ptr<Features> data)
{
	// init features with data if necessary and assure type is correct
	if (data)
	{
		if (!data->has_property(FP_DOT))
				error("Specified features are not of type CDotFeatures");
		set_features(data);
	}

	auto dotdata=data->as<DotFeatures>();
	m_mean=dotdata->get_mean();
	SGMatrix<float64_t> cov=dotdata->get_cov();
	decompose_cov(cov);
	init();
	return true;
}

int32_t Gaussian::get_num_model_parameters()
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

float64_t Gaussian::get_log_model_parameter(int32_t num_param)
{
	not_implemented(SOURCE_LOCATION);
	return 0;
}

float64_t Gaussian::get_log_derivative(int32_t num_param, int32_t num_example)
{
	not_implemented(SOURCE_LOCATION);
	return 0;
}

float64_t Gaussian::get_log_likelihood_example(int32_t num_example)
{
	ASSERT(features->has_property(FP_DOT))
	SGVector<float64_t> v=features->as<DotFeatures>()->get_computed_dot_feature_vector(num_example);
	float64_t answer=compute_log_PDF(v);
	return answer;
}

float64_t Gaussian::update_params_em(const SGVector<float64_t> alpha_k)
{
	auto dotdata=features->as<DotFeatures>();
	int32_t num_dim=dotdata->get_dim_feature_space();

	// compute mean
	float64_t alpha_k_sum=0;
	SGVector<float64_t> mean(num_dim);
	linalg::zero(mean);

	for (auto i: range(alpha_k.vlen))
	{
		alpha_k_sum+=alpha_k[i];
		SGVector<float64_t> v=dotdata->get_computed_dot_feature_vector(i);
		linalg::add(v, mean, mean, alpha_k[i], 1.0);
	}

	linalg::scale(mean, mean, 1.0 / alpha_k_sum);

	set_mean(mean);

	// compute covariance matrix

	SGMatrix<float64_t> cov_sum;
	ECovType cov_type=get_cov_type();
	if (cov_type==FULL)
	{
		cov_sum = SGMatrix<float64_t>(num_dim, num_dim);
		cov_sum.zero();
	}
	else if(cov_type==DIAG)
	{
		cov_sum = SGMatrix<float64_t>(1, num_dim);
		cov_sum.zero();
	}
	else if(cov_type==SPHERICAL)
	{
		cov_sum = SGMatrix<float64_t>(1, 1);
		cov_sum.zero();
	}

	for (auto j: range(alpha_k.vlen))
	{
		SGVector<float64_t> v=dotdata->get_computed_dot_feature_vector(j);
		linalg::add(v, mean, v, -1.0, 1.0);

		switch (cov_type)
		{
		case FULL:
#ifdef HAVE_LAPACK
			cblas_dger(
			    CblasRowMajor, num_dim, num_dim, alpha_k[j], v.vector, 1,
			    v.vector, 1, (double*)cov_sum.matrix, num_dim);
#else
			linalg::rank_update(cov_sum, v, alpha_k[j]);
#endif
			break;
		case DIAG:
			for (int32_t k = 0; k < num_dim; k++)
				cov_sum(1, k) += v.vector[k] * v.vector[k] * alpha_k[j];

			break;
		case SPHERICAL:
			float64_t temp = 0;

			temp = linalg::dot(v, v);

			cov_sum(0, 0) += temp * alpha_k[j];
			break;
		}
	}

	switch (cov_type)
	{
	case FULL:
	{
		linalg::scale(cov_sum, cov_sum, 1 / alpha_k_sum);

		SGVector<float64_t> d0(num_dim);
#ifdef HAVE_LAPACK
		d0.vector = SGMatrix<float64_t>::compute_eigenvectors(
		    cov_sum.matrix, num_dim, num_dim);
#else
		// FIXME use eigenvectors computeation warpper by micmn
		typename SGMatrix<float64_t>::EigenMatrixXtMap eig = cov_sum;
		typename SGVector<float64_t>::EigenVectorXtMap eigenvalues_eig = d0;

		Eigen::EigenSolver<typename SGMatrix<float64_t>::EigenMatrixXt> solver(
		    eig);
		eigenvalues_eig = solver.eigenvalues().real();
#endif

		set_d(d0);
		set_u(cov_sum);

		break;
	}
	case DIAG:
		linalg::scale(cov_sum, cov_sum, 1 / alpha_k_sum);

		set_d(cov_sum.get_row_vector(0));

		break;

	case SPHERICAL:
		cov_sum[0] /= alpha_k_sum * num_dim;

		set_d(cov_sum.get_row_vector(0));

		break;
	}

	return alpha_k_sum;
}

float64_t Gaussian::compute_log_PDF(SGVector<float64_t> point)
{
	ASSERT(m_mean.vector && m_d.vector)
	ASSERT(point.vlen == m_mean.vlen)
	SGVector<float64_t> difference = point.clone();

	linalg::add(difference, m_mean, difference, -1.0, 1.0);

	float64_t answer=m_constant;

	if (m_cov_type==FULL)
	{
		SGVector<float64_t> temp_holder(m_d.vlen);
		temp_holder.zero();
#ifdef HAVE_LAPACK
		cblas_dgemv(
		    CblasRowMajor, CblasNoTrans, m_d.vlen, m_d.vlen, 1, m_u.matrix,
		    m_d.vlen, difference, 1, 0, temp_holder, 1);
#else
		linalg::dgemv<float64_t>(1, m_u, false, difference, 0, temp_holder);
#endif

		for (int32_t i=0; i<m_d.vlen; i++)
			answer+=temp_holder[i]*temp_holder[i]/m_d.vector[i];
	}
	else if (m_cov_type==DIAG)
	{
		for (int32_t i=0; i<m_mean.vlen; i++)
			answer+=difference[i]*difference[i]/m_d.vector[i];
	}
	else
	{
		for (int32_t i=0; i<m_mean.vlen; i++)
			answer += difference[i] * difference[i] / m_d.vector[0];
	}

	return -0.5 * answer;
}

SGVector<float64_t> Gaussian::get_mean()
{
	return m_mean;
}

void Gaussian::set_mean(SGVector<float64_t> mean)
{
	if (mean.vlen==1)
		m_cov_type=SPHERICAL;

	m_mean=mean;
}

void Gaussian::set_cov(SGMatrix<float64_t> cov)
{
	ASSERT(cov.num_rows==cov.num_cols)
	ASSERT(cov.num_rows==m_mean.vlen)
	decompose_cov(cov);
	init();
}

void Gaussian::set_d(const SGVector<float64_t> d)
{
	m_d = d;
	init();
}

SGMatrix<float64_t> Gaussian::get_cov()
{
	SGMatrix<float64_t> cov(m_mean.vlen, m_mean.vlen);

	if (m_cov_type==FULL)
	{
		if (!m_u.matrix)
			error("Unitary matrix not set");

		SGMatrix<float64_t> temp_holder(m_mean.vlen, m_mean.vlen);
		SGMatrix<float64_t> diag_holder(m_mean.vlen, m_mean.vlen);
		for (int32_t i = 0; i < m_d.vlen; i++)
			diag_holder(i, i) = m_d.vector[i];
#ifdef HAVE_LAPACK
		cblas_dgemm(
		    CblasRowMajor, CblasTrans, CblasNoTrans, m_d.vlen, m_d.vlen,
		    m_d.vlen, 1, m_u.matrix, m_d.vlen, diag_holder.matrix, m_d.vlen, 0,
		    temp_holder.matrix, m_d.vlen);
		cblas_dgemm(
		    CblasRowMajor, CblasNoTrans, CblasNoTrans, m_d.vlen, m_d.vlen,
		    m_d.vlen, 1, temp_holder.matrix, m_d.vlen, m_u.matrix, m_d.vlen, 0,
		    cov.matrix, m_d.vlen);
#else
		linalg::dgemm<float64_t>(
		    1, m_u, diag_holder, true, false, 0, temp_holder);
		linalg::dgemm<float64_t>(1, temp_holder, m_u, false, false, 0, cov);
#endif
	}
	else if (m_cov_type == DIAG)
	{
		for (int32_t i = 0; i < m_d.vlen; i++)
			cov(i, i) = m_d.vector[i];
	}
	else
	{
		for (int32_t i = 0; i < m_mean.vlen; i++)
			cov(i, i) = m_d.vector[0];
	}
	return cov;
}

void Gaussian::register_params()
{
	SG_ADD(&m_u, "m_u", "Unitary matrix.");
	SG_ADD(&m_d, "m_d", "Diagonal.");
	SG_ADD(&m_mean, "m_mean", "Mean.");
	SG_ADD(&m_constant, "m_constant", "Constant part.");
	SG_ADD_OPTIONS(
	    (machine_int_t*)&m_cov_type, "m_cov_type", "Covariance type.",
	    ParameterProperties::NONE, SG_OPTIONS(FULL, DIAG, SPHERICAL));
}

void Gaussian::decompose_cov(SGMatrix<float64_t> cov)
{
	switch (m_cov_type)
	{
	case FULL:
	{
		m_u = SGMatrix<float64_t>(cov.num_rows, cov.num_rows);
		m_u = cov.clone();
		m_d = SGVector<float64_t>(cov.num_rows);
#ifdef HAVE_LAPACK
		m_d.vector = SGMatrix<float64_t>::compute_eigenvectors(
		    m_u.matrix, cov.num_rows, cov.num_rows);
#else
		// FIXME use eigenvectors computeation warpper by micmn
		typename SGMatrix<float64_t>::EigenMatrixXtMap eig = m_u;
		typename SGVector<float64_t>::EigenVectorXtMap eigenvalues_eig = m_d;

		Eigen::EigenSolver<typename SGMatrix<float64_t>::EigenMatrixXt> solver(
		    eig);
		eigenvalues_eig = solver.eigenvalues().real();
#endif
		break;
	}
	case DIAG:
		m_d = SGVector<float64_t>(cov.num_rows);
		for (int32_t i = 0; i < cov.num_rows; i++)
			m_d[i] = cov.matrix[i * cov.num_rows + i];

		break;
	case SPHERICAL:
		m_d = SGVector<float64_t>(1);
		m_d.vector[0] = cov.matrix[0];
		break;
	}
}

SGVector<float64_t> Gaussian::sample()
{
	SG_TRACE("Entering");
	SGMatrix<float64_t> r_matrix(m_mean.vlen, m_mean.vlen);
	r_matrix.zero();

	switch (m_cov_type)
	{
	case FULL:
	case DIAG:
		for (int32_t i = 0; i < m_mean.vlen; i++)
			r_matrix(i, i) = std::sqrt(m_d.vector[i]);

		break;
	case SPHERICAL:
		for (int32_t i = 0; i < m_mean.vlen; i++)
			r_matrix(i, i) = std::sqrt(m_d.vector[0]);

		break;
	}

	SGVector<float64_t> random_vec(m_mean.vlen);
	random::fill_array(random_vec, NormalDistribution<float64_t>(), m_prng);

	if (m_cov_type == FULL)
	{
		SGMatrix<float64_t> temp_matrix(m_d.vlen, m_d.vlen);
		temp_matrix.zero();
#ifdef HAVE_LAPACK
		cblas_dgemm(
		    CblasRowMajor, CblasNoTrans, CblasNoTrans, m_d.vlen, m_d.vlen,
		    m_d.vlen, 1, m_u.matrix, m_d.vlen, r_matrix.matrix, m_d.vlen, 0,
		    temp_matrix.matrix, m_d.vlen);
#else
		linalg::dgemm<float64_t>(
		    1, m_u, r_matrix, false, false, 0, temp_matrix);
#endif
		r_matrix = temp_matrix;
	}

	SGVector<float64_t> samp(m_mean.vlen);

#ifdef HAVE_LAPACK
	cblas_dgemv(
	    CblasRowMajor, CblasNoTrans, m_mean.vlen, m_mean.vlen, 1,
	    r_matrix.matrix, m_mean.vlen, random_vec.vector, 1, 0, samp.vector, 1);
#else
	linalg::dgemv<float64_t>(1.0, r_matrix, false, random_vec, 0.0, samp);
#endif
	for (int32_t i = 0; i < m_mean.vlen; i++)
		samp.vector[i] += m_mean.vector[i];

	SG_TRACE("Leaving");
	return samp;
}

std::shared_ptr<Gaussian> Gaussian::obtain_from_generic(std::shared_ptr<Distribution> distribution)
{
	if (!distribution)
		return NULL;

	auto casted=std::dynamic_pointer_cast<Gaussian>(distribution);
	if (!casted)
		return NULL;

	/* since an additional reference is returned */

	return casted;
}
