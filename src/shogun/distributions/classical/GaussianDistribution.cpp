/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#ifdef HAVE_EIGEN3

#include <shogun/distributions/classical/GaussianDistribution.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CGaussianDistribution::CGaussianDistribution() : CProbabilityDistribution()
{
	init();
}

CGaussianDistribution::CGaussianDistribution(SGVector<float64_t> mean,
		SGMatrix<float64_t> cov,
		ECovarianceFactorization factorization, bool cov_is_factor) :
				CProbabilityDistribution(mean.vlen)
{
	REQUIRE(cov.num_rows==cov.num_cols, "Covariance must be square but is "
			"%dx%d\n", cov.num_rows, cov.num_cols);
	REQUIRE(mean.vlen==cov.num_cols, "Mean must have same dimension as "
			"covariance, which is %dx%d, but is %d\n",
			cov.num_rows, cov.num_cols, mean.vlen);

	init();

	m_mean=mean;
	m_factorization=factorization;

	if (!cov_is_factor)
		compute_covariance_factorization(cov, factorization);
	else
		m_L=cov;
}

CGaussianDistribution::~CGaussianDistribution()
{

}

SGMatrix<float64_t> CGaussianDistribution::sample(int32_t num_samples,
		SGMatrix<float64_t> pre_samples) const
{
	REQUIRE(num_samples>0, "Number of samples must be positive, but is %d\n",
			num_samples);

	/* use pre-allocated samples? */
	SGMatrix<float64_t> samples;
	if (pre_samples.matrix)
	{
		REQUIRE(pre_samples.num_rows==m_dimension, "Dimension of pre-samples"
				" (%d) does not match dimension of Gaussian (%d)\n",
				pre_samples.num_rows, m_dimension);

		REQUIRE(pre_samples.num_cols==num_samples, "Number of pre-samples"
				" (%d) does not desired number of samples (%d)\n",
				pre_samples.num_cols, num_samples);

		samples=pre_samples;
	}
	else
	{
		/* allocate memory and sample from std normal */
		samples=SGMatrix<float64_t>(m_dimension, num_samples);
		for (index_t i=0; i<m_dimension*num_samples; ++i)
			samples.matrix[i]=sg_rand->std_normal_distrib();
	}

	/* map into desired Gaussian covariance */
	Map<MatrixXd> eigen_samples(samples.matrix, samples.num_rows,
			samples.num_cols);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	eigen_samples=eigen_L*eigen_samples;

	/* add mean */
	Map<VectorXd> eigen_mean(m_mean.vector, m_mean.vlen);
	eigen_samples.colwise()+=eigen_mean;

	return samples;
}

SGVector<float64_t> CGaussianDistribution::log_pdf(SGMatrix<float64_t> samples) const
{
	REQUIRE(samples.num_cols>0, "Number of samples must be positive, but is %d\n",
			samples.num_cols);
	REQUIRE(samples.num_rows=m_dimension, "Sample dimension (%d) does not match"
			"Gaussian dimension (%d)\n", samples.num_rows, m_dimension);

	/* for easier to read code */
	index_t num_samples=samples.num_cols;

	float64_t const_part=-0.5 * m_dimension * CMath::log(2 * CMath::PI);

	/* log-determinant */
	float64_t log_det_part=0;
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	switch (m_factorization)
	{
		case CF_CHOLESKY:
		{
			/* determinant is product of diagonal elements of triangular matrix */
			VectorXd diag=eigen_L.diagonal();
			log_det_part=-diag.array().log().sum();
			break;
		}
		case CF_SVD_QR:
		{
			/* use QR for computing log-determinant, which abs(log-det R),
			 * note that since covariance is psd, determinant is positive, so
			 * absolute value here is fine (and necessary, since determinant
			 * of R might be negative. */
			Map<MatrixXd> eigen_R(m_R.matrix, m_R.num_rows, m_R.num_cols);
			VectorXd diag=eigen_R.diagonal();
			log_det_part=-0.5*diag.array().abs().log().sum();
			break;
		}
	}

	/* sample based part */
	Map<MatrixXd> eigen_samples(samples.matrix, samples.num_rows,
			samples.num_cols);
	Map<VectorXd> eigen_mean(m_mean.vector, m_mean.vlen);

	/* substract mean from samples (create copy) */
	SGMatrix<float64_t> centred(m_dimension, num_samples);
	Map<MatrixXd> eigen_centred(centred.matrix, centred.num_rows,
			centred.num_cols);
	for (index_t dim=0; dim<m_dimension; ++dim)
	{
		for (index_t sample_idx=0; sample_idx<num_samples; ++sample_idx)
			centred(dim,sample_idx)=samples(dim,sample_idx)-m_mean[dim];
	}

	/* solve the linear system based on factorization */
	MatrixXd solved;
	switch (m_factorization)
	{
		case CF_CHOLESKY:
		{
			/* use triangular solver */
			solved=eigen_L.triangularView<Lower>().solve(eigen_centred);
			solved=eigen_L.transpose().triangularView<Upper>().solve(solved);
			break;
		}
		case CF_SVD_QR:
		{
			/* use orthogonality of Q and triangular solver */
			Map<MatrixXd> eigen_Q(m_Q.matrix, m_Q.num_rows, m_Q.num_cols);
			Map<MatrixXd> eigen_R(m_R.matrix, m_R.num_rows, m_R.num_cols);

			solved=eigen_Q.transpose()*eigen_centred;
			solved=eigen_R.triangularView<Upper>().solve(solved);
			break;
		}
	}

	/* one quadratic part x^T C^-1 x for each sample x */
	SGVector<float64_t> result(num_samples);
	Map<VectorXd> eigen_result(result.vector, result.vlen);
	for (index_t i=0; i<num_samples; ++i)
	{
		/* i-th centred sample */
		VectorXd left=eigen_centred.block(0, i, m_dimension, 1);

		/* inverted covariance times i-th centred sample */
		VectorXd right=solved.block(0,i,m_dimension,1);
		result[i]=-0.5*left.dot(right);
	}

	/* combine and return */
	eigen_result=eigen_result.array()+(log_det_part+const_part);

	/* contains everything */
	return result;

}

void CGaussianDistribution::init()
{
	m_factorization=CF_CHOLESKY;

	SG_ADD(&m_mean, "mean", "Mean of the Gaussian.", MS_NOT_AVAILABLE);
	SG_ADD(&m_L, "L", "Lower factor of covariance matrix, "
			"depending on the factorization type.", MS_NOT_AVAILABLE);
	SG_ADD(&m_Q, "Q", "Orthogonal Q factor of QR of covariance, if used.",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_R, "R", "Triangular R factor of QR of covariance, if used.",
			MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*)&m_factorization, "factorization", "Type of the "
			"factorization of the covariance matrix.", MS_NOT_AVAILABLE);
}

void CGaussianDistribution::compute_covariance_factorization(
		SGMatrix<float64_t> cov, ECovarianceFactorization factorization)
{
	Map<MatrixXd> eigen_cov(cov.matrix, cov.num_rows, cov.num_cols);
	m_L=SGMatrix<float64_t>(cov.num_rows, cov.num_cols);
	Map<MatrixXd> eigen_factor(m_L.matrix, m_L.num_rows, m_L.num_cols);

	switch (m_factorization)
	{
		case CF_CHOLESKY:
		{
			LLT<MatrixXd> llt(eigen_cov);
			if (llt.info()==NumericalIssue)
				SG_ERROR("Error computing Cholesky\n");

			eigen_factor=llt.matrixL();
			break;
		}
		case CF_SVD_QR:
		{
			JacobiSVD<MatrixXd> svd(eigen_cov, ComputeFullU);
			MatrixXd U=svd.matrixU();
			VectorXd s=svd.singularValues();

			/* square root of covariance using all eigenvectors */
			eigen_factor=U.array().rowwise()*s.transpose().array().sqrt();

			/* QR factorization of covariance for log-pdf */
			m_Q=SGMatrix<float64_t>(cov.num_rows, cov.num_cols);
			m_R=SGMatrix<float64_t>(cov.num_rows, cov.num_cols);
			Map<MatrixXd> eigen_Q(m_Q.matrix, m_Q.num_rows, m_Q.num_cols);
			Map<MatrixXd> eigen_R(m_R.matrix, m_R.num_rows, m_R.num_cols);

			ColPivHouseholderQR<MatrixXd> qr(eigen_cov);
			eigen_Q=qr.householderQ();
			eigen_R=qr.matrixQR().triangularView<Upper>();
			break;
		}
		default:
			SG_ERROR("Unknown factorization type: %d\n", m_factorization);
			break;
	}
}

#endif // HAVE_EIGEN3
