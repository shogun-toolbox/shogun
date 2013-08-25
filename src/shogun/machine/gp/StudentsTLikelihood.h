/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 * Copyright (C) 2013 Roman Votyakov
 *
 * Code adapted from the GPML Toolbox:
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 */

#ifndef CSTUDENTSTLIKELIHOOD_H_
#define CSTUDENTSTLIKELIHOOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/LikelihoodModel.h>

namespace shogun
{

/** @brief Class that models a Student's-t likelihood.
 *
 * \f[
 * p(y|f)=\prod_{i=1}^{n} \frac{\Gamma(\frac{\nu+1}{2})}
 * {\Gamma(\frac{\nu}{2})\sqrt{\nu\pi}\sigma}
 * \left(1+\frac{(y_i-f_i)^2}{\nu\sigma^2} \right)^{-\frac{\nu+1}{2}}
 * \f]
 *
 * The hyperparameters of the Student's t-likelihood model are \f$\sigma\f$ -
 * scale parameter, and \f$\nu\f$ - degrees of freedom.
 */
class CStudentsTLikelihood: public CLikelihoodModel
{
public:
	/** default constructor */
	CStudentsTLikelihood();

	/** constructor
	 *
	 * @param sigma noise variance
	 * @param df degrees of freedom
	 */
	CStudentsTLikelihood(float64_t sigma, float64_t df);

	virtual ~CStudentsTLikelihood();

	/** returns the name of the likelihood model
	 *
	 * @return name StudentsTLikelihood
	 */
	virtual const char* get_name() const { return "StudentsTLikelihood"; }

	/** returns the scale paramter
	 *
	 * @return scale parameter
	 */
	float64_t get_sigma() { return m_sigma; }

	/** sets the scale parameter
	 *
	 * @param sigma scale parameter
	 */
	void set_sigma(float64_t sigma)
	{
		REQUIRE(sigma>0.0, "Scale parameter must be greater than zero\n")
		m_sigma=sigma;
	}

	/** get degrees of freedom
	 *
	 * @return degrees of freedom
	 */
	float64_t get_degrees_freedom() { return m_df; }

	/** set degrees of freedom
	 *
	 * @param df degrees of freedom
	 */
	void set_degrees_freedom(float64_t df)
	{
		REQUIRE(df>1.0, "Number of degrees of freedom must be greater than one\n")
		m_df=df;
	}

	/** helper method used to specialize a base class instance
	 *
	 * @param likelihood likelihood model
	 * @return casted CStudentsTLikelihood object
	 */
	static CStudentsTLikelihood* obtain_from_generic(CLikelihoodModel* likelihood);

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
	 * @return model type Student's-t
	 */
	virtual ELikelihoodModelType get_model_type() const { return LT_STUDENTST; }

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

	/** returns the zeroth moment of a given (unnormalized) probability
	 * distribution:
	 *
	 * \f[
	 * log(Z_i) = log\left(\int p(y_i|f_i) \mathcal{N}(f_i|\mu,\sigma^2)
	 * df_i\right)
	 * \f]
	 *
	 * for each \f$f_i\f$.
	 *
	 * @param mu mean of the \f$\mathcal{N}(f_i|\mu,\sigma^2)\f$
	 * @param s2 variance of the \f$\mathcal{N}(f_i|\mu,\sigma^2)\f$
	 * @param lab labels \f$y_i\f$
	 *
	 * @return log zeroth moments \f$log(Z_i)\f$
	 */
	virtual SGVector<float64_t> get_log_zeroth_moments(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab) const;

	/** returns the first moment of a given (unnormalized) probability
	 * distribution \f$q(f_i) = Z_i^-1
	 * p(y_i|f_i)\mathcal{N}(f_i|\mu,\sigma^2)\f$, where \f$ Z_i=\int
	 * p(y_i|f_i)\mathcal{N}(f_i|\mu,\sigma^2) df_i\f$.
	 *
	 * This method is useful for EP local likelihood approximation.
	 *
	 * @param mu mean of the \f$\mathcal{N}(f_i|\mu,\sigma^2)\f$
	 * @param s2 variance of the \f$\mathcal{N}(f_i|\mu,\sigma^2)\f$
	 * @param lab labels \f$y_i\f$
	 * @param i index i
	 *
	 * @return first moment of \f$q(f_i)\f$
	 */
	virtual float64_t get_first_moment(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab, index_t i) const;

	/** returns the second moment of a given (unnormalized) probability
	 * distribution \f$q(f_i) = Z_i^-1
	 * p(y_i|f_i)\mathcal{N}(f_i|\mu,\sigma^2)\f$, where \f$ Z_i=\int
	 * p(y_i|f_i)\mathcal{N}(f_i|\mu,\sigma^2) df_i\f$.
	 *
	 * This method is useful for EP local likelihood approximation.
	 *
	 * @param mu mean of the \f$\mathcal{N}(f_i|\mu,\sigma^2)\f$
	 * @param s2 variance of the \f$\mathcal{N}(f_i|\mu,\sigma^2)\f$
	 * @param lab labels \f$y_i\f$
	 * @param i index i
	 *
	 * @return the second moment of \f$q(f_i)\f$
	 */
	virtual float64_t get_second_moment(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab, index_t i) const;

	/** return whether Student's likelihood function supports regression
	 *
	 * @return true
	 */
	virtual bool supports_regression() const { return true; }

private:
	/** initialize function */
	void init();

	/** scale parameter */
	float64_t m_sigma;

	/** degrees of freedom */
	float64_t m_df;
};
}
#endif /* HAVE_EIGEN3 */
#endif /* CSTUDENTSTLIKELIHOOD_H_ */
