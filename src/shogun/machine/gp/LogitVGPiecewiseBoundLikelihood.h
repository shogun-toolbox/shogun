/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
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
 * Code adapted from
 * https://github.com/emtiyaz/VariationalApproxExample
 * and the reference paper is
 * Marlin, Benjamin M., Mohammad Emtiyaz Khan, and Kevin P. Murphy.
 * "Piecewise Bounds for Estimating Bernoulli-Logistic Latent Gaussian Models." ICML. 2011.
 */

#ifndef _LOGITVGPIECEWISEBOUNDLIKELIHOOD_H_
#define _LOGITVGPIECEWISEBOUNDLIKELIHOOD_H_

#include <shogun/lib/config.h>


#include <shogun/machine/gp/LogitLikelihood.h>
#include <shogun/machine/gp/VariationalGaussianLikelihood.h>

namespace shogun
{
/** @brief Class that models Logit likelihood and
 * uses variational piecewise bound to approximate
 * the following variational expection of log likelihood
 * \f[
 * \sum_{{i=1}^n}{E_{q(f_i|{\mu}_i,{\sigma}^2_i)}[logP(y_i|f_i)]}
 * \f] where
 * \f[
 * p(y_i|f_i) = \frac{exp(y_i*f_i)}{1+exp(f_i)}, y_i \in \{0,1\}
 * \f]
 *
 * The memory requirement for this class is O(n*m), where
 * n is the size of sample, which is the size of mu or sigma2 passed in set_distribution
 * m is the size of the pre-defined bound, which is the num_rows of bound passed in set_bound
 * (In the reference Matlab code, m is 20)
 */
class LogitVGPiecewiseBoundLikelihood : public VariationalGaussianLikelihood
{
public:
	LogitVGPiecewiseBoundLikelihood();

	~LogitVGPiecewiseBoundLikelihood() override;

	/** returns the name of the likelihood model
	 *
	 * @return name LogitVGPiecewiseBoundLikelihood
	 */
	const char* get_name() const override { return "LogitVGPiecewiseBoundLikelihood"; }

	/** set the variational piecewise bound for logit likelihood
	 *
	 *  @param bound variational piecewise bound
	 */
	virtual void set_variational_bound(SGMatrix<float64_t> bound);

	/** set the variational normal distribution given data and parameters
	 *
	 * @param mu mean of the variational normal distribution
	 * @param s2 variance of the variational normal distribution
	 * @param lab labels/data used
	 * @return true if variational parameters are valid
	 *
	 */
	bool set_variational_distribution(SGVector<float64_t> mu, SGVector<float64_t> s2, std::shared_ptr<const Labels> lab) override;

	/** returns the expection of the logarithm of a logit distribution
	 * wrt the variational distribution using piecewise bound
	 *
	 * For each sample i, using the piecewise bound to
	 * approximate \f[
	 * E_{q(f_i|{\mu}_i,{\sigma}^2_i)}[logP(y_i|f_i)]
	 * \f] given mu_i and sigma2_i
	 *
	 * @return expection
	 */
	SGVector<float64_t> get_variational_expection() override;

#ifndef SWIG
	/** get derivative of the variational expection of log LogitLikelihood
	 * using the piecewise bound with respect to given parameter
	 *
	 * compute the derivative of \f[
	 * E_{q(f_i|{\mu}_i,{\sigma}^2_i)}[logP(y_i|f_i)]
	 * \f] given mu_i and sigma2_i with repect to param using the piecewise bound
	 *
	 * @param param parameter(mu or sigma2)
	 *
	 * @return derivative
	 */
	SGVector<float64_t> get_variational_first_derivative(Parameters::const_reference param) const override;
#endif

	/** return whether likelihood function supports
	 * computing the derivative wrt hyperparameter
	 * Note that variational parameters are NOT considered as hyperparameters
	 *
	 * @return boolean
	 */
	bool supports_derivative_wrt_hyperparameter() const override { return false; }

#ifndef SWIG
	/** get derivative of log likelihood \f$log(p(y|f))\f$ with respect to given
	 * hyperparameter
	 * Note that variational parameters are NOT considered as hyperparameters
	 *
	 * @param param parameter
	 *
	 * @return derivative
	 */
	SGVector<float64_t> get_first_derivative_wrt_hyperparameter(Parameters::const_reference param) const override;
#endif

	/** initialize the default bound for this class */
	void set_default_variational_bound();

protected:

	/** The function used to initialize m_likelihood*/
	void init_likelihood() override;

private:
	/** initialize private data members for this class */
	void init();

	/** compute common variables later used in get_variational_expection
	 * and get_variational_first_derivative.
	 * Note that this method will automatically be called when set_variational_distribution is called
	 */
	void precompute();

	/** Variational piecewise bound for logit likelihood */
	SGMatrix<float64_t> m_bound;

	/** The pdf given the lower range and parameters(mu and variance) */
	SGMatrix<float64_t> m_pl;

	/** The pdf given the higher range and parameters(mu and variance) */
	SGMatrix<float64_t> m_ph;

	/** The CDF difference between the lower and higher range given the parameters(mu and variance) */
	SGMatrix<float64_t> m_cdf_diff;

	/** The result of l^2 + sigma^2 */
	SGMatrix<float64_t> m_l2_plus_s2;

	/** The result of h^2 + sigma^2 */
	SGMatrix<float64_t> m_h2_plus_s2;

	/** The result of l*pdf(l_norm)-h*pdf(h_norm) */
	SGMatrix<float64_t> m_weighted_pdf_diff;
};
}
#endif /* _LOGITVGPIECEWISEBOUNDLIKELIHOOD_H_ */
