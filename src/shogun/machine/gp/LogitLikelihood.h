/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#ifndef _LOGITLIKELIHOOD_H_
#define _LOGITLIKELIHOOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/LikelihoodModel.h>

namespace shogun
{

/** @brief Class that models Logit likelihood.
 *
 * \f[
 * p(y|f) = \prod_{i=1}^n \frac{1}{1+exp(-y_i*f_i)}
 * \f]
 */
class CLogitLikelihood : public CLikelihoodModel
{
public:
	/** default constructor */
	CLogitLikelihood();

	virtual ~CLogitLikelihood();

	/** returns the name of the likelihood model
	 *
	 * @return name LogitLikelihood
	 */
	virtual const char* get_name() const { return "LogitLikelihood"; }

	/** returns the logarithm of the predictive density of \f$y_*\f$:
	 *
	 * \f[
	 * log(p(y_*|X,y,x_*)) = log\left(\int p(y_*|f_*) p(f_*|X,y,x_*) df_*\right)
	 * \f]
	 *
	 * which approximately equals to
	 *
	 * \f[
	 * log\left(\int p(y_*|f_*) \mathcal{N}(f*|\mu,\sigma^2) df*\right)
	 * \f]
	 *
	 * where normal distribution \f$\mathcal{N}(\mu,\sigma^2)\f$ is an
	 * approximation to the posterior marginal \f$p(f_*|X,y,x_*)\f$.
	 *
	 * NOTE: if lab equals to NULL, then each \f$y_*\f$ equals to one.
	 *
	 * @param mu posterior mean of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param s2 posterior variance of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param lab labels \f$y_*\f$
	 *
	 * @return \f$log(p(y_*|X, y, x*))\f$ for each label \f$y_*\f$
	 */
	virtual SGVector<float64_t> get_predictive_log_probabilities(
			SGVector<float64_t> mu, SGVector<float64_t> s2,
			const CLabels* lab=NULL) const;

	/** returns mean of the predictive marginal \f$p(y_*|X,y,x_*)\f$.
	 *
	 * NOTE: if lab equals to NULL, then each \f$y_*\f$ equals to one.
	 *
	 * @param mu posterior mean of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param s2 posterior variance of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param lab labels \f$y_*\f$
	 *
	 * @return final means evaluated by likelihood function
	 */
	virtual SGVector<float64_t> get_predictive_means(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab=NULL) const;

	/** returns variance of the predictive marginal \f$p(y_*|X,y,x_*)\f$.
	 *
	 * NOTE: if lab equals to NULL, then each \f$y_*\f$ equals to one.
	 *
	 * @param mu posterior mean of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param s2 posterior variance of a Gaussian distribution
	 * \f$\mathcal{N}(\mu,\sigma^2)\f$, which is an approximation to the
	 * posterior marginal \f$p(f_*|X,y,x_*)\f$
	 * @param lab labels \f$y_*\f$
	 *
	 * @return final variances evaluated by likelihood function
	 */
	virtual SGVector<float64_t> get_predictive_variances(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab=NULL) const;

	/** get model type
	 *
	 * @return model type Logit
	 */
	virtual ELikelihoodModelType get_model_type() const { return LT_LOGIT; }

	/** returns the logarithm of the point-wise likelihood \f$log(p(y_i|f_i))\f$
	 * for each label \f$y_i\f$.
	 *
	 * One can evaluate log-likelihood like: \f$log(p(y|f)) = \sum_{i=1}^{n}
	 * log(p(y_i|f_i))\f$
	 *
	 * @param lab labels \f$y_i\f$
	 * @param func values of the function \f$f_i\f$
	 *
	 * @return logarithm of the point-wise likelihood
	 */
	virtual SGVector<float64_t> get_log_probability_f(const CLabels* lab,
			SGVector<float64_t> func) const;

	/** get derivative of log likelihood \f$log(P(y|f))\f$ with respect to
	 * function location \f$f\f$
	 *
	 * @param lab labels used
	 * @param func function location
	 * @param i index, choices are 1, 2, and 3 for first, second, and third
	 * derivatives respectively
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_log_probability_derivative_f(
			const CLabels* lab, SGVector<float64_t> func, index_t i) const;

	/** get derivative of log likelihood \f$log(P(y|f))\f$ with respect to given
	 * parameter
	 *
	 * @param lab labels used
	 * @param param parameter
	 * @param func function location
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_first_derivative(const CLabels* lab,
			const TParameter* param, SGVector<float64_t> func) const;

	/** get derivative of the first derivative of log likelihood with respect to
	 * function location, i.e. \f$\frac{\partial log(P(y|f))}{\partial f}\f$
	 * with respect to given parameter
	 *
	 * @param lab labels used
	 * @param param parameter
	 * @param func function location
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_second_derivative(const CLabels* lab,
			const TParameter* param, SGVector<float64_t> func) const;

	/** get derivative of the second derivative of log likelihood with respect
	 * to function location, i.e. \f$\frac{\partial^{2} log(P(y|f))}{\partial
	 * f^{2}}\f$ with respect to given parameter
	 *
	 * @param lab labels used
	 * @param param parameter
	 * @param func function location
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_third_derivative(const CLabels* lab,
			const TParameter* param, SGVector<float64_t> func) const;

	/** return whether logit likelihood function supports binary classification
	 *
	 * @return true
	 */
	virtual bool supports_binary() const { return true; }
};
}
#endif /* HAVE_EIGEN3 */
#endif /* _LOGITLIKELIHOOD_H_ */
