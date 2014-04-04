/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
 * Written (W) 2013 Heiko Strathmann
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 */
#ifdef HAVE_EIGEN3

#ifndef GAUSSIANDISTRIBUTION_H
#define GAUSSIANDISTRIBUTION_H

#include <shogun/lib/config.h>
#include <shogun/distributions/classical/ProbabilityDistribution.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{


/** @brief Dense version of the well-known Gaussian probability distribution,
 * defined as
 * \f[
 * \mathcal{N}_x(\mu,\Sigma)=
 * \frac{1}{\sqrt{|2\pi\Sigma|}}
 * \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)
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


	/** Computes the univariate pdf for one given sample.
	 *
	 * @param sample is a given sample 
	 * @param mu is the mean of univariate Normal distribution (default value is 0.0)
	 * @param sigma2 is the variance of univariate Normal distribution (default value is 1.0)
	 * @return the pdf of the distribution given the sample
	 */
	static float64_t univariate_log_pdf(float64_t sample, float64_t mu = 0.0, float64_t sigma2 = 1.0)
	{
		REQUIRE(sigma2 > 0, "Variance should be positive\n");
		return -0.5 * (CMath::pow(sample - mu, 2) / sigma2
			+ CMath::log(2.0 * CMath::PI) + CMath::log(sigma2));
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
