/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Roman Votyakov, Viktor Gal
 */

#include <shogun/distributions/classical/GaussianDistribution.h>


#include <shogun/base/Parameter.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/mathematics/RandomNamespace.h>

using namespace shogun;
using namespace Eigen;

GaussianDistribution::GaussianDistribution() : RandomMixin<ProbabilityDistribution>()
{
	init();
}

GaussianDistribution::GaussianDistribution(SGVector<float64_t> mean,
		SGMatrix<float64_t> cov, bool cov_is_factor) :
				RandomMixin<ProbabilityDistribution>(mean.vlen)
{
	require(cov.num_rows==cov.num_cols, "Covariance must be square but is "
			"{}x{}", cov.num_rows, cov.num_cols);
	require(mean.vlen==cov.num_cols, "Mean must have same dimension as "
			"covariance, which is {}x{}, but is {}",
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
				error("Error computing Cholesky of Gaussian's covariance. "
						"Smallest Eigenvalue is {}.", ev[0]);
			}
		}

		eigen_L=llt.matrixL();
	}
	else
		m_L=cov;
}

GaussianDistribution::~GaussianDistribution()
{

}

SGMatrix<float64_t> GaussianDistribution::sample(int32_t num_samples,
		SGMatrix<float64_t> pre_samples) const
{
	require(num_samples>0, "Number of samples ({}) must be positive",
			num_samples);

	/* use pre-allocated samples? */
	SGMatrix<float64_t> samples;
	if (pre_samples.matrix)
	{
		require(pre_samples.num_rows==m_dimension, "Dimension of pre-samples"
				" ({}) does not match dimension of Gaussian ({})",
				pre_samples.num_rows, m_dimension);

		require(pre_samples.num_cols==num_samples, "Number of pre-samples"
				" ({}) does not desired number of samples ({})",
				pre_samples.num_cols, num_samples);

		samples=pre_samples;
	}
	else
	{
		/* allocate memory and sample from std normal */
		samples=SGMatrix<float64_t>(m_dimension, num_samples);
		random::fill_array(samples, NormalDistribution<float64_t>(), m_prng);
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

SGVector<float64_t> GaussianDistribution::log_pdf_multiple(SGMatrix<float64_t> samples) const
{
	require(samples.num_cols>0, "Number of samples must be positive, but is {}",
			samples.num_cols);
	require(samples.num_rows==m_dimension, "Sample dimension ({}) does not match"
			"Gaussian dimension ({})", samples.num_rows, m_dimension);

	/* for easier to read code */
	index_t num_samples=samples.num_cols;

	float64_t const_part = -0.5 * m_dimension * std::log(2 * Math::PI);

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

void GaussianDistribution::init()
{
	SG_ADD(&m_mean, "mean", "Mean of the Gaussian.");
	SG_ADD(&m_L, "L", "Lower factor of covariance matrix, "
			"depending on the factorization type.");
}
