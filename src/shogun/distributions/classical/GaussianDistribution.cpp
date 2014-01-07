/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#ifdef HAVE_EIGEN3

#include <distributions/classical/GaussianDistribution.h>
#include <base/Parameter.h>
#include <mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CGaussianDistribution::CGaussianDistribution() : CProbabilityDistribution()
{
	init();
}

CGaussianDistribution::CGaussianDistribution(SGVector<float64_t> mean,
		SGMatrix<float64_t> cov, bool cov_is_factor) :
				CProbabilityDistribution(mean.vlen)
{
	REQUIRE(cov.num_rows==cov.num_cols, "Covariance must be square but is "
			"%dx%d\n", cov.num_rows, cov.num_cols);
	REQUIRE(mean.vlen==cov.num_cols, "Mean must have same dimension as "
			"covariance, which is %dx%d, but is %d\n",
			cov.num_rows, cov.num_cols, mean.vlen);

	init();

	m_mean=mean;

	if (!cov_is_factor)
	{
		Map<MatrixXd> eigen_cov(cov.matrix, cov.num_rows, cov.num_cols);
		m_L=SGMatrix<float64_t>(cov.num_rows, cov.num_cols);
		Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

		/* compute cholesky */
		LLT<MatrixXd> llt(eigen_cov);
		if (llt.info()==NumericalIssue)
		{
			/* try to compute smalles eigenvalue for information */
			SelfAdjointEigenSolver<MatrixXd> solver(eigen_cov);
			if (solver.info() == Success)
			{
				VectorXd ev=solver.eigenvalues();
				SG_ERROR("Error computing Cholesky of Gaussian's covariance. "
						"Smallest Eigenvalue is %f.\n", ev[0]);
			}
		}

		eigen_L=llt.matrixL();
	}
	else
		m_L=cov;
}

CGaussianDistribution::~CGaussianDistribution()
{

}

SGMatrix<float64_t> CGaussianDistribution::sample(int32_t num_samples,
		SGMatrix<float64_t> pre_samples) const
{
	REQUIRE(num_samples>0, "Number of samples (%d) must be positive\n",
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

SGVector<float64_t> CGaussianDistribution::log_pdf_multiple(SGMatrix<float64_t> samples) const
{
	REQUIRE(samples.num_cols>0, "Number of samples must be positive, but is %d\n",
			samples.num_cols);
	REQUIRE(samples.num_rows=m_dimension, "Sample dimension (%d) does not match"
			"Gaussian dimension (%d)\n", samples.num_rows, m_dimension);

	/* for easier to read code */
	index_t num_samples=samples.num_cols;

	float64_t const_part=-0.5 * m_dimension * CMath::log(2 * CMath::PI);

	/* determinant is product of diagonal elements of triangular matrix */
	float64_t log_det_part=0;
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	VectorXd diag=eigen_L.diagonal();
	log_det_part=-diag.array().log().sum();

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
	MatrixXd solved=eigen_L.triangularView<Lower>().solve(eigen_centred);
	solved=eigen_L.transpose().triangularView<Upper>().solve(solved);

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
	SG_ADD(&m_mean, "mean", "Mean of the Gaussian.", MS_NOT_AVAILABLE);
	SG_ADD(&m_L, "L", "Lower factor of covariance matrix, "
			"depending on the factorization type.", MS_NOT_AVAILABLE);
}

#endif // HAVE_EIGEN3
