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

#ifndef _LOGITVGLIKELIHOOD_H_
#define _LOGITVGLIKELIHOOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/machine/gp/VariationalGaussianLikelihood.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/machine/gp/LogitLikelihood.h>

namespace shogun
{
template<class C> class SGMatrix;

/** @brief Class that models Logit likelihood and 
 * uses numerical integration to approximate 
 * the following variational expection of log likelihood
 * \f[
 * \sum_{{i=1}^n}{E_{q(f_i|{\mu}_i,{\sigma}^2_i)}[logP(y_i|f_i)]}
 * \f] where
 * \f[
 * p(y_i|f_i) = \frac{exp(y_i*f_i)}{1+exp(f_i)}, y_i \in \{0,1\}
 * \f]
 *
 */
class CLogitVGLikelihood : public CVariationalGaussianLikelihood
{
public:
	CLogitVGLikelihood();

	virtual ~CLogitVGLikelihood();


	/** returns the name of the likelihood model
	 *
	 * @return name LogitVGLikelihood
	 */
	virtual const char* get_name() const { return "LogitVGLikelihood"; }

	/** set the variational normal distribution given data and parameters
	 *
	 * @param mu mean of the variational normal distribution
	 * @param s2 variance of the variational normal distribution
	 * @param lab labels/data used
	 *
	 */
	virtual void set_variational_distribution(SGVector<float64_t> mu,
		SGVector<float64_t> s2, const CLabels* lab);

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
	virtual SGVector<float64_t> get_variational_expection();

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
	virtual SGVector<float64_t> get_variational_first_derivative(const TParameter* param) const;


	// The following methods will be subjected to change
	virtual ELikelihoodModelType get_model_type() const { return LT_LOGIT; }

	virtual SGVector<float64_t> get_predictive_means(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab=NULL) const
	{ return likelihood->get_predictive_means(mu, s2, lab); }

	virtual SGVector<float64_t> get_predictive_variances(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab=NULL) const
	{ return likelihood->get_predictive_variances(mu, s2, lab); }

	virtual SGVector<float64_t> get_log_probability_f(const CLabels* lab,
			SGVector<float64_t> func) const
	{ return likelihood->get_log_probability_f(lab, func); }

	virtual SGVector<float64_t> get_log_probability_derivative_f(
			const CLabels* lab, SGVector<float64_t> func, index_t i) const
	{ return likelihood->get_log_probability_derivative_f(lab, func, i); }

	virtual bool supports_binary() const { return likelihood->supports_binary(); }

	virtual SGVector<float64_t> get_log_zeroth_moments(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab) const
	{ return likelihood->get_log_zeroth_moments(mu, s2, lab); }

	virtual float64_t get_first_moment(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab, index_t i) const
	{ return likelihood->get_first_moment(mu, s2, lab, i); }

	virtual float64_t get_second_moment(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab, index_t i) const
	{ return likelihood->get_second_moment(mu, s2, lab, i); }
private:

	CLogitLikelihood * likelihood;

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

	/** The mean of variational normal distribution */
	SGVector<float64_t> m_mu;

	/** The variance of variational normal distribution */
	SGVector<float64_t> m_s2;

	SGVector<float64_t> m_lab;
};
}
#endif /* HAVE_EIGEN3 */
#endif /* _LOGITVGLIKELIHOOD_H_ */
