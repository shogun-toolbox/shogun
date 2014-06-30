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
 */

#ifndef _DUALVARIATIONALGAUSSIANLIKELIHOOD_H_ 
#define _DUALVARIATIONALGAUSSIANLIKELIHOOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#include <shogun/machine/gp/VariationalGaussianLikelihood.h>

namespace shogun
{
/** @brief Class that models dual variational likelihood 
 *
 * This likelihood model is described in the reference paper
 * Mohammad Emtiyaz Khan, Aleksandr Y. Aravkin, Michael P. Friedlander, Matthias Seeger
 * Fast Dual Variational Inference for Non-Conjugate Latent Gaussian Models. ICML2013
 *
 * The mathematical definition (equation 19 in the paper) is as below
 * \f[
 * Fenchel_i(\alpha_i,\lambda_i) = max_{h_i,\rho_i}{\alpha_i h_i+\lambda_i \rho_i /2 - E_{q(f_i|h_i,\rho_i)}(-log(p(y_i|f_i)))}
 * \f]
 * where \f$\alpha_i\f$,\f$\lambda_i\f$ are Lagrange multipliers with respective to constraints
 * \f$h_i=\mu_i\f$ and \f$\rho_i=\sigma_i^2\f$ respectively,
 * \f$\mu\f$ and \f$\sigma_i\f$ are variational Gaussian parameters,
 * y_i is data label, \f$q(f_i)\f$ is the variational Gaussian distribution,
 * and p(y_i) is the data distribution to be specified.
 * In this setting, \f$\alpha\f$ and \f$\lambda\f$ are called dual parameters for \f$\mu\f$ and \f$\sigma^2\f$ respectively.
 *
 */
class CDualVariationalGaussianLikelihood : public CVariationalGaussianLikelihood
{
public:
	/** default constructor */
	CDualVariationalGaussianLikelihood();

	virtual ~CDualVariationalGaussianLikelihood();

	/** returns the name of the likelihood model
	 *
	 * @return name DualVariationalGaussianLikelihood
	 */
	virtual const char* get_name() const { return "DualVariationalGaussianLikelihood"; }

	/** returns the expection of the logarithm of a given probability distribution 
	 * wrt the variational distribution given m_mu and m_s2
	 *
	 * @return expection
	 */
	virtual SGVector<float64_t> get_variational_expection();

	/** get derivative of the variational expection of log likelihood 
	 * with respect to given parameter
	 *
	 * @param param parameter
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_variational_first_derivative(const TParameter* param) const;

	/** return whether likelihood function supports
	 * computing the derivative wrt hyperparameter
	 * Note that variational parameters are NOT considered as hyperparameters
	 *
	 * @return boolean
	 */
	virtual bool supports_derivative_wrt_hyperparameter() const;

	/** get derivative of log likelihood \f$log(p(y|f))\f$ with respect to given
	 * hyperparameter
	 * Note that variational parameters are NOT considered as hyperparameters
	 *
	 * @param param parameter
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_first_derivative_wrt_hyperparameter(const TParameter* param) const;

	/** set the variational distribution given data and parameters
	 *
	 * @param mu mean of the variational distribution
	 * @param s2 variance of the variational distribution
	 * @param lab labels/data used
	 *
	 * Note that the variational distribution is Gaussian
	 */
	virtual void set_variational_distribution(SGVector<float64_t> mu,
		SGVector<float64_t> s2, const CLabels* lab);

	/** check whether the dual parameters are valid or not.
	 * 
	 *  @return true if dual parameters are valid
	 */
	virtual bool dual_parameters_valid() const;

	/** this method is used for adjusting step size
	 * to ensure the updated value satisfied lower/upper bound constrain  
	 *
	 * The updated value is defined as below.
	 * lambda_new = m_lambda + direction * step
	 *
	 * @param direction direction for m_lambda update
	 * @param step original step size (non-negative)
	 * @return adjusted step size
	 */
	virtual float64_t adjust_step_wrt_dual_parameter(SGVector<float64_t> direction, const float64_t step) const;

	/** set dual parameters for variational parameters
	 *
	 * @param lambda dual parameter for variational mean
	 * @param lab labels/data used
	 *
	 * Note that dual parameter (alpha) for the variational variance
	 * is implicitly set based on lambda
	 */
	virtual void set_dual_parameters(SGVector<float64_t> lambda,  const CLabels* lab);

	/** get the dual parameter (alpha) for variational mu
	 *
	 * @return the dual parameter (alpha)
	 */
	virtual SGVector<float64_t> get_mu_dual_parameter() const=0;

	/** get the dual parameter (lambda) for variational s2
	 *
	 * @return the dual parameter (lambda)
	 */
	virtual SGVector<float64_t> get_variance_dual_parameter() const=0;

	/** get the upper bound for dual parameter (lambda)
	 *
	 * @return the upper bound
	 */
	virtual float64_t get_dual_upper_bound() const=0;

	/** get the lower bound for dual parameter (lambda)
	 *
	 * @return the lower bound
	 */
	virtual float64_t get_dual_lower_bound() const=0;

	/** whether the upper bound is strict
	 *
	 * @return true if the upper bound is strict
	 */
	virtual bool dual_upper_bound_strict() const=0;

	/** whether the lower bound is strict
	 *
	 * @return true if the lower bound is strict
	 */
	virtual bool dual_lower_bound_strict() const=0;

	/** evaluate the dual objective function
	 *
	 * @return the value of Fenchel conjugates given m_lambda
	 */
	virtual SGVector<float64_t> get_dual_objective_value()=0;

	/** get the derivative of the dual objective function with respect to param
	 *
	 * @param param parameter
	 * @return the value of of the derivative
	 *
	 */
	virtual SGVector<float64_t> get_dual_first_derivative(const TParameter* param) const=0;

	/** set the m_strict_scale
	 * 
	 * @param strict_scale must be between 0 and 1 exclusively
	 *
	 */
	virtual void set_strict_scale(float64_t strict_scale);
protected:

	/** The dual variables (lambda) for the variational parameter s2.
	 *
	 * Note that in variational Gaussian inference, there is a relationship
	 * between lambda and alpha, where alpha is the dual parameter for variational parameter mu
	 *
	 * Therefore, the dual variables (alpha) for variational parameter mu is not explicitly saved.
	 */
	SGVector<float64_t> m_lambda;

	/** The value used to ensure strict bound(s) for m_lambda in adjust_step_wrt_dual_parameter() 
	 * 
	 * Note that the value should be between 0 and 1 exclusively. 
	 *
	 * The default value is 1e-5.
	 */
	float64_t m_strict_scale;

	/** whether m_lambda is satisfied lower bound and/or upper bound condition. */
	bool m_is_valid;

	/** compute common variables later used in get_variational_expection
	 * and get_variational_first_derivative.
	 * Note that this method will automatically be called when set_variational_distribution is called
	 */
	virtual void precompute();

	/** this method is used to dynamic-cast the likelihood model, m_likelihood,
	 * to variational likelihood model.
	 */
	virtual CVariationalGaussianLikelihood* get_variational_likelihood() const;
private:
	/** initialize private data members for this class */
	void init();

};
}
#endif /* HAVE_EIGEN3 */
#endif /* _DUALVARIATIONALGAUSSIANLIKELIHOOD_H_ */
