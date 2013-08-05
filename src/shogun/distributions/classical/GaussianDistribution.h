/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */
#ifdef HAVE_EIGEN3

#ifndef GAUSSIANDISTRIBUTION_H
#define GAUSSIANDISTRIBUTION_H

#include <shogun/distributions/classical/ProbabilityDistribution.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{


/** @brief Dense version of the well-known Gaussian probability distribution,
 * defined as
 * \f[
 * \mathcal{N}_x(\mu,\Sigma)=
 * \frac{1}{\sqrt{|2\pi\Sigma|}}
 * \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu\right)
 * \f]
 *
 * The implementation represents the covariance matrix \f$\Sigma \f$, as
 * Cholesky factorisation, such that the covariance can be computed as
 * \f$\Sigma=LL^T\f$.
 */

class CGaussianDistribution: public CProbabilityDistribution
{
public:
	/** Default constructor */
	CGaussianDistribution();

	/** Constructor for which takes Gaussian mean and its covariance matrix.
	 * It is also possible to pass a precomputed matrix factor of the specified
	 * form. In this case, the factorization is not explicitly computed.
	 *
	 * @param mean mean of the Gaussian
	 * @param cov covariance of the Gaussian, or covariance factor
	 * @param cov_is_factor whether cov is a factor of the covariance or not
	 * (default is false). If false, the factorization is explicitly computed
	 */
	CGaussianDistribution(SGVector<float64_t> mean, SGMatrix<float64_t> cov,
			bool cov_is_factor=false);

	/** Destructor */
	virtual ~CGaussianDistribution();

	/** Samples from the distribution multiple times
	 *
	 * @param num_samples number of samples to generate
	 * @param pre_samples a matrix of standard normal samples that might be used
	 * for sampling the Gaussian. Ignored by default. If passed, the pre-samples
	 * will be modified.
	 * @return matrix with samples (column vectors)
	 */
	virtual SGMatrix<float64_t> sample(int32_t num_samples,
			SGMatrix<float64_t> pre_samples=SGMatrix<float64_t>()) const;

	/** Computes the log-pdf for all provided samples. That is
	 *
	 * \f[
	 * \log(\mathcal{N}_x(\mu,\Sigma))=
	 * - \frac{d}{2}  \log(2\pi)
	 * -\frac{1}{2}\log(\det(\Sigma))
	 * -\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu),
	 * \f]
	 *
	 * where \f$d\f$ is the dimension of the Gaussian.
	 * The method to compute the log-determinant is based on the factorization
	 * of the covariance matrix.
	 *
	 * @param samples samples to compute log-pdf of (column vectors)
	 * @return vector with log-pdfs of given samples
	 */
	virtual SGVector<float64_t> log_pdf_multiple(SGMatrix<float64_t> samples) const;

	/** @return name of the SGSerializable */
	virtual const char* get_name() const
	{
		return "GaussianDistribution";
	}

private:

	/** Initialses and registers parameters */
	void init();

protected:
	/** Mean */
	SGVector<float64_t> m_mean;

	/** Lower factor of covariance matrix (depends on factorization type).
	 * Covariance (approximation) is given by \f$\Sigma=LL^T\f$ */
	SGMatrix<float64_t> m_L;
};

}

#endif // GAUSSIANDISTRIBUTION_H
#endif // HAVE_EIGEN3
