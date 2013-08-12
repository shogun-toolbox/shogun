/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 * Copyright (C) 2013 Roman Votyakov
 */

#ifndef CGAUSSIANLIKELIHOOD_H_
#define CGAUSSIANLIKELIHOOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/LikelihoodModel.h>

namespace shogun
{

/** @brief Class that models Gaussian likelihood.
 *
 * \f[
 * p(y|f)=\prod_{i=1}^n\frac{1} {\sqrt{2\pi\sigma^2}}
 * exp\left(-\frac{(y_i-f_i)^2}{2\sigma^2}\right)
 * \f]
 *
 * The hyperparameter of the Gaussian likelihood model is standard deviation:
 * \f$\sigma\f$.
 */
class CGaussianLikelihood: public CLikelihoodModel
{
public:
	/** default constructor */
	CGaussianLikelihood();

	/** constructor
	 *
	 * @param sigma observation noise
	 */
	CGaussianLikelihood(float64_t sigma);

	virtual ~CGaussianLikelihood();

	/** returns the name of the likelihood model
	 *
	 * @return name GaussianLikelihood
	 */
	virtual const char* get_name() const { return "GaussianLikelihood"; }

	/** returns the noise standard deviation
	 *
	 * @return noise standard deviation
	 */
	float64_t get_sigma() { return m_sigma; }

	/** sets the noise standard deviation
	 *
	 * @param sigma noise standard deviation
	 */
	void set_sigma(float64_t sigma)
	{
		REQUIRE(sigma>0.0, "Standard deviation must be greater than zero\n")
		m_sigma=sigma;
	}

	/** helper method used to specialize a base class instance
	 *
	 * @param lik likelihood model
	 * @return casted CGaussianLikelihood object
	 */
	static CGaussianLikelihood* obtain_from_generic(CLikelihoodModel* lik);

	/** returns the logarithm of the predictive density of \f$y_*\f$:
	 *
	 * \f[
	 * log(p(y_*|X,y,x_*)) = log\left(\int p(y_*|f_*)
	 * p(f_*|X,y,x_*) df_*\right)
	 * \f]
	 *
	 * which approximately equals to
	 *
	 * \f[
	 * log\left(\int p(y_*|f_*) N(f*|\mu,\sigma^2) df*\right)
	 * \f]
	 *
	 * where normal distribution \f$N(\mu,\sigma^2)\f$ is an approximation to
	 * the posterior marginal \f$p(f_*|X,y,x_*)\f$
	 *
	 * NOTE: if lab equals to NULL, then each \f$y_*\f$ equals to one.
	 *
	 * @param mu posterior mean of a Gaussian distribution
	 * \f$N(\mu,\sigma^2)\f$, which is an approximation to the posterior
	 * marginal \f$p(f_*|X,y,x_*)\f$
	 * @param s2 posterior variance of a Gaussian distribution
	 * \f$N(\mu,\sigma^2)\f$, which is an approximation to the posterior
	 * marginal \f$p(f_*|X,y,x_*)\f$
	 * @param lab labels \f$y_*\f$. NOTE: if lab equals to NULL, then each
	 * \f$y_*\f$ equals to one.
	 *
	 * @return \f$log(p(y_*|X, y, x*))\f$ for each label \f$y_*\f$
	 */
	virtual SGVector<float64_t> evaluate_log_probabilities(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab=NULL) const;

	/** returns mean of the predictive marginal \f$p(y_*|X,y,x_*)\f$
	 *
	 * NOTE: if lab equals to NULL, then each \f$y_*\f$ equals to one.
	 *
	 * @param mu posterior mean of a Gaussian distribution
	 * \f$N(\mu,\sigma^2)\f$, which is an approximation to the posterior
	 * marginal \f$p(f_*|X,y,x_*)\f$
	 * @param s2 posterior variance of a Gaussian distribution
	 * \f$N(\mu,\sigma^2)\f$, which is an approximation to the posterior
	 * marginal \f$p(f_*|X,y,x_*)\f$
	 * @param lab labels \f$y_*\f$
	 *
	 * @return final means evaluated by likelihood function
	 */
	virtual SGVector<float64_t> evaluate_means(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab=NULL) const;

	/** returns variance of the predictive marginal \f$p(y_*|X,y,x_*)\f$
	 *
	 * NOTE: if lab equals to NULL, then each \f$y_*\f$ equals to one.
	 *
	 * @param mu posterior mean of a Gaussian distribution
	 * \f$N(\mu,\sigma^2)\f$, which is an approximation to the posterior
	 * marginal \f$p(f_*|X,y,x_*)\f$
	 * @param s2 posterior variance of a Gaussian distribution
	 * \f$N(\mu,\sigma^2)\f$, which is an approximation to the posterior
	 * marginal \f$p(f_*|X,y,x_*)\f$
	 * @param lab labels \f$y_*\f$
	 *
	 * @return final variances evaluated by likelihood function
	 */
	virtual SGVector<float64_t> evaluate_variances(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab=NULL) const;

	/** get model type
	 *
	 * @return model type Gaussian
	 */
	virtual ELikelihoodModelType get_model_type() const { return LT_GAUSSIAN; }

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

	/** return whether Gaussian likelihood function supports regression
	 *
	 * @return true
	 */
	virtual bool supports_regression() const { return true; }

private:
	/** initialize function */
	void init();

	/** standard deviation */
	float64_t m_sigma;
};
}
#endif /* HAVE_EIGEN3 */
#endif /* CGAUSSIANLIKELIHOOD_H_ */
