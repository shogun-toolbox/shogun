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
 * http://hannes.nickisch.org/code/approxXX.tar.gz
 * and the reference paper is
 * Nickisch, Hannes, and Carl Edward Rasmussen.
 * "Approximations for Binary Gaussian Process Classification."
 * Journal of Machine Learning Research 9.10 (2008).
 */

#ifndef _NUMERICALVGLIKELIHOOD_H_
#define _NUMERICALVGLIKELIHOOD_H_

#include <shogun/lib/config.h>

#include <shogun/machine/gp/VariationalGaussianLikelihood.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/RegressionLabels.h>


namespace shogun
{
template<class C> class SGMatrix;

/** @brief Class that models likelihood and
 * uses numerical integration to approximate
 * the following variational expection of log likelihood
 * \f[
 * \sum_{{i=1}^n}{E_{q(f_i|{\mu}_i,{\sigma}^2_i)}[logP(y_i|f_i)]}
 * \f]
 */
class NumericalVGLikelihood : public VariationalGaussianLikelihood
{
public:
	NumericalVGLikelihood();

	~NumericalVGLikelihood() override;

	/** returns the name of the likelihood model
	 *
	 * @return name NumericalVGLikelihood
	 */
	const char* get_name() const override { return "NumericalVGLikelihood"; }

	/** set the variational Gaussian distribution given data and parameters
	 *
	 * @param mu mean of the variational Gaussian distribution
	 * @param s2 variance of the variational Gaussian distribution
	 * @param lab labels/data used
	 * @return true if variational parameters are valid
	 *
	 */
	bool set_variational_distribution(SGVector<float64_t> mu,
		SGVector<float64_t> s2, std::shared_ptr<const Labels> lab) override;

	/** returns the expection of the logarithm of a logit distribution
	 * wrt the variational distribution using numerical integration
	 *
	 * For each sample i, using Gaussian-Hermite quadrature to
	 * approximate \f[
	 * E_{q(f_i|{\mu}_i,{\sigma}^2_i)}[logP(y_i|f_i)]
	 * \f] given mu_i and sigma2_i
	 *
	 * @return expection
	 */
	SGVector<float64_t> get_variational_expection() override;

#ifndef SWIG
	/** get derivative of the variational expection of log LogitLikelihood
	 * using numerical integration with respect to given parameter
	 *
	 * compute the derivative of \f[
	 * E_{q(f_i|{\mu}_i,{\sigma}^2_i)}[logP(y_i|f_i)]
	 * \f] given mu_i and sigma2_i with repect to param using Gaussian-Hermite quadrature
	 *
	 * @param param parameter(mu or sigma2)
	 *
	 * @return derivative
	 */
	SGVector<float64_t> get_variational_first_derivative(Parameters::const_reference param) const override;

	/** get derivative of log likelihood \f$log(p(y|f))\f$ with respect to given
	 * hyperparameter
	 * Note that variational parameters (mu and sigma) are NOT considered as hyperparameters
	 *
	 * @param param parameter
	 *
	 * @return derivative
	 */
	SGVector<float64_t> get_first_derivative_wrt_hyperparameter(Parameters::const_reference param) const override;
#endif

	/** set the number of Gaussian Hermite point used to compute variational expection
	 *
	 * @param n number of Gaussian Hermite point
	 *
	 * The default value is 20.
	 */
	virtual void set_GHQ_number(index_t n);

protected:

	/** The function used to initialize m_likelihood defined in VariationalLikelihood
	 * Note that for some compiler removing this line will issue an error
	 * */
	void init_likelihood() override =0;

private:

	/** Using N Gaussian-Hermite quadrature points */
	index_t m_GHQ_N;

	/** whether Gaussian-Hermite quadrature points are are initialized or not */
	bool m_is_init_GHQ;

	/** initialize private data members for this class */
	void init();

	/** compute common variables later used in get_variational_expection
	 * and get_variational_first_derivative.
	 * Note that this method will automatically be called when set_variational_distribution is called
	 */
	void precompute();

	/** Gaussian-Hermite quadrature base points (abscissas) for logit likelihood */
	SGVector<float64_t> m_xgh;

	/** Gaussian-Hermite quadrature weight factors for logit likelihood */
	SGVector<float64_t> m_wgh;

	/** The result of used for computing variational expection */
	SGMatrix<float64_t> m_log_lam;
};
}
#endif /* _NUMERICALVGLIKELIHOOD_H_ */
