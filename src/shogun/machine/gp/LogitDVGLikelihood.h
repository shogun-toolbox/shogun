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
 * the reference paper is
 * Mohammad Emtiyaz Khan, Aleksandr Y. Aravkin, Michael P. Friedlander, Matthias Seeger
 * Fast Dual Variational Inference for Non-Conjugate Latent Gaussian Models. ICML2013
 *

 */

#ifndef _LOGITDVGLIKELIHOOD_H_
#define _LOGITDVGLIKELIHOOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/LogitLikelihood.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/machine/gp/DualVariationalGaussianLikelihood.h>

namespace shogun
{
/** @brief Class that models dual variational logit likelihood 
 *
 * This likelihood model is described in the reference paper
 * Mohammad Emtiyaz Khan, Aleksandr Y. Aravkin, Michael P. Friedlander, Matthias Seeger
 * Fast Dual Variational Inference for Non-Conjugate Latent Gaussian Models. ICML2013
 *
 * The mathematically definition (equation 19 in the paper) is as below
 * \f[
 * Fenchel_i(\alpha_i,\lambda_i) = max_{h_i,\rho_i}{\alpha_i h_i+\lambda_i \rho_i /2 - E_{q(f_i|h_i,\rho_i)}(-log(p(y_i|f_i)))}
 * \f]
 * where \f$\alpha_i\f$,\f$\lambda_i\f$ are Lagrange multipliers with respective to constraints
 * \f$h_i=\mu_i\f$ and \f$\rho_i=\sigma_i^2\f$ respectively,
 * \f$\mu\f$ and \f$\sigma_i\f$ are variational Gaussian parameters,
 * \f$y_i\f$ is data label, \f$q(f_i)\f$ is the variational Gaussian distribution,
 * and \f$p(y_i)\f$ is the data distribution to be specified.
 * In this setting, \f$\alpha\f$ and \f$\lambda\f$ are called dual parameters for \f$\mu\f$ and \f$\sigma^2\f$ respectively.
 *
 * Note that \f$p(y_i)\f$ is Logistic distribution and a local variational bound defined as below is used to approximate
 * -E_{q(f_i|h_i,\rho_i)}(-log(p(y_i|f_i)))
 *
 * The local variational bound used here is
 * \f[
 * log(x) \leq t^{-1}x+log(t)-1 
 * \f], where t is a local variable and the inequality holds for every t>0. 
 * See Bernoulli-logit in Table 2 of the paper for detailed information
 */
class CLogitDVGLikelihood : public CDualVariationalGaussianLikelihood
{
public:
	CLogitDVGLikelihood();

	virtual ~CLogitDVGLikelihood();

	/** returns the name of the likelihood model
	 *
	 * @return name LogitDVGLikelihood
	 */
	virtual const char* get_name() const { return "LogitDVGLikelihood"; }

	/** evaluate the dual objective function
	 *
	 * @return the value of Fenchel conjugates given m_lambda
	 */
	virtual SGVector<float64_t> get_dual_objective_value();

	/** get the derivative of the dual objective function with respect to param
	 *
	 * @param param parameter
	 * @return the value of of the derivative
	 *
	 */
	virtual SGVector<float64_t> get_dual_first_derivative(const TParameter* param) const;

	/** get the upper bound for dual parameter (lambda)
	 *
	 * @return the upper bound
	 */
	virtual float64_t get_dual_upper_bound() const{return 1.0;};

	/** get the lower bound for dual parameter (lambda)
	 *
	 * @return the lower bound
	 */
	virtual float64_t get_dual_lower_bound() const{return 0.0;};

	/** whether the upper bound is strict
	 *
	 * @return true if the upper bound is strict
	 */
	virtual bool is_dual_upper_bound_strict() const {return true;};

	/** whether the lower bound is strict
	 *
	 * @return true if the lower bound is strict
	 */
	virtual bool is_dual_lower_bound_strict() const {return true;};

	/** get the dual parameter (alpha) for variational mu
	 *
	 * Note that alpha = m_lambda - label
	 * For detailed information, please refer to the paper.
	 *
	 * @return the dual parameter (alpha)
	 */
	virtual SGVector<float64_t> get_mu_dual_parameter() const;

	/** get the dual parameter (lambda) for variational s2
	 *
	 * @return the dual parameter (lambda)
	 */
	virtual SGVector<float64_t> get_variance_dual_parameter() const;

protected:
	/** The function used to initialize m_likelihood*/
	virtual void init_likelihood();

private:
	/** initialize private data members for this class */
	void init();

};
}
#endif /* HAVE_EIGEN3 */
#endif /* _LOGITDVGLIKELIHOOD_H_ */
