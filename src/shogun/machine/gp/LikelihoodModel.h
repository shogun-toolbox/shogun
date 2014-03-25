/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Written (W) 2013 Heiko Strathmann
 * Copyright (C) 2012 Jacob Walker
 * Copyright (C) 2013 Roman Votyakov
 */

#ifndef CLIKELIHOODMODEL_H_
#define CLIKELIHOODMODEL_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/labels/Labels.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{

/** type of likelihood model */
enum ELikelihoodModelType
{
	LT_NONE=0,
	LT_GAUSSIAN=10,
	LT_STUDENTST=20,
	LT_LOGIT=30,
	LT_PROBIT=40,
	LT_SOFTMAX=50
};

/** @brief The Likelihood model base class.
 *
 * The Likelihood model computes approximately the distribution \f$p(y|f)\f$,
 * where \f$y\f$ are the labels, and \f$f\f$ is the prediction function.
 */
class CLikelihoodModel : public CSGObject
{
public:
	/** default constructor */
	CLikelihoodModel();

	virtual ~CLikelihoodModel();

	/** returns the logarithm of the predictive density of \f$y_*\f$:
	 *
	 * \f[
	 * log(p(y_*|X,y,x_*)) = log\left(\int p(y_*|f_*) p(f_*|X,y,x_*) df_*\right)
	 * \f]
	 *
	 * which approximately equals to
	 *
	 * \f[
	 * log\left(\int p(y_*|f_*) \mathcal{N}(f_*|\mu,\sigma^2) df_*\right)
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
			const CLabels* lab=NULL);

	/** returns mean of the predictive marginal \f$p(y_*|X,y,x_*)\f$
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
			SGVector<float64_t> s2, const CLabels* lab=NULL) const=0;

	/** returns variance of the predictive marginal \f$p(y_*|X,y,x_*)\f$
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
			SGVector<float64_t> s2, const CLabels* lab=NULL) const=0;

	/** get model type
	  *
	  * @return model type NONE
	 */
	virtual ELikelihoodModelType get_model_type() const { return LT_NONE; }

	/** Returns the logarithm of the point-wise likelihood \f$log(p(y_i|f_i))\f$
	 * for each label \f$y_i\f$.
	 *
	 * One can evaluate log-likelihood like: \f$ log(p(y|f)) = \sum_{i=1}^{n}
	 * log(p(y_i|f_i))\f$
	 *
	 * @param lab labels \f$y_i\f$
	 * @param func values of the function \f$f_i\f$
	 *
	 * @return logarithm of the point-wise likelihood
	 */
	virtual SGVector<float64_t> get_log_probability_f(const CLabels* lab,
			SGVector<float64_t> func) const=0;

	/** Returns the log-likelihood \f$log(p(y|f)) = \sum_{i=1}^{n}
	 * log(p(y_i|f_i))\f$ for each of the provided functions \f$ f \f$ in the
	 * given matrix.
	 *
	 * Wrapper method which calls get_log_probability_f multiple times.
	 *
	 * @param lab labels \f$y_i\f$
	 * @param F values of the function \f$f_i\f$ where each column of the matrix
	 * is one function \f$ f \f$.
	 *
	 * @return log-likelihood for every provided function
	 */
	virtual SGVector<float64_t> get_log_probability_fmatrix(const CLabels* lab,
			SGMatrix<float64_t> F) const;

	/** get derivative of log likelihood \f$log(p(y|f))\f$ with respect to
	 * location function \f$f\f$
	 *
	 * @param lab labels used
	 * @param func function location
	 * @param i index, choices are 1, 2, and 3 for first, second, and third
	 * derivatives respectively
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_log_probability_derivative_f(
			const CLabels* lab, SGVector<float64_t> func, index_t i) const=0;

	/** get derivative of log likelihood \f$log(p(y|f))\f$ with respect to given
	 * parameter
	 *
	 * @param lab labels used
	 * @param func function location
	 * @param param parameter
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_first_derivative(const CLabels* lab,
			SGVector<float64_t> func, const TParameter* param) const
	{
		SG_ERROR("Can't compute derivative wrt %s parameter\n", param->m_name)
		return SGVector<float64_t>();
	}


	/** get derivative of the first derivative of log likelihood with respect to
	 * function location, i.e. \f$\frac{\partial log(p(y|f))}{\partial f}\f$
	 * with respect to given parameter
	 *
	 * @param lab labels used
	 * @param func function location
	 * @param param parameter
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_second_derivative(const CLabels* lab,
			SGVector<float64_t> func, const TParameter* param) const
	{
		SG_ERROR("Can't compute derivative wrt %s parameter\n", param->m_name)
		return SGVector<float64_t>();
	}

	/** get derivative of the second derivative of log likelihood with respect
	 * to function location, i.e. \f$\frac{\partial^{2} log(p(y|f))}{\partial
	 * f^{2}}\f$ with respect to given parameter
	 *
	 * @param lab labels used
	 * @param func function location
	 * @param param parameter
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_third_derivative(const CLabels* lab,
			SGVector<float64_t> func, const TParameter* param) const
	{
		SG_ERROR("Can't compute derivative wrt %s parameter\n", param->m_name)
		return SGVector<float64_t>();
	}

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
	 * @return log zeroth moment \f$log(Z_i)\f$
	 */
	virtual SGVector<float64_t> get_log_zeroth_moments(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab) const=0;

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
			SGVector<float64_t> s2, const CLabels* lab, index_t i) const=0;

	/** returns the first moment of a given (unnormalized) probability
	 * distribution \f$q(f_i) = Z_i^-1
	 * p(y_i|f_i)\mathcal{N}(f_i|\mu,\sigma^2)\f$ for each \f$f_i\f$, where \f$
	 * Z_i=\int p(y_i|f_i)\mathcal{N}(f_i|\mu,\sigma^2) df_i\f$.
	 *
	 * Wrapper method which calls get_first_moment multiple times.
	 *
	 * @param mu mean of the \f$\mathcal{N}(f_i|\mu,\sigma^2)\f$
	 * @param s2 variance of the \f$\mathcal{N}(f_i|\mu,\sigma^2)\f$
	 * @param lab labels \f$y_i\f$
	 *
	 * @return the first moment of \f$q(f_i)\f$ for each \f$f_i\f$
	 */
	virtual SGVector<float64_t> get_first_moments(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab) const;

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
			SGVector<float64_t> s2, const CLabels* lab, index_t i) const=0;

	/** returns the second moment of a given (unnormalized) probability
	 * distribution \f$q(f_i) = Z_i^-1
	 * p(y_i|f_i)\mathcal{N}(f_i|\mu,\sigma^2)\f$ for each \f$f_i\f$, where \f$
	 * Z_i=\int p(y_i|f_i)\mathcal{N}(f_i|\mu,\sigma^2) df_i\f$.
	 *
	 * Wrapper method which calls get_second_moment multiple times.
	 *
	 * @param mu mean of the \f$\mathcal{N}(f_i|\mu,\sigma^2)\f$
	 * @param s2 variance of the \f$\mathcal{N}(f_i|\mu,\sigma^2)\f$
	 * @param lab labels \f$y_i\f$
	 *
	 * @return the second moment of \f$q(f_i)\f$ for each \f$f_i\f$
	 */
	virtual SGVector<float64_t> get_second_moments(SGVector<float64_t> mu,
			SGVector<float64_t> s2, const CLabels* lab) const;

	/** return whether likelihood function supports regression
	 *
	 * @return false
	 */
	virtual bool supports_regression() const { return false; }

	/** return whether likelihood function supports binary classification
	 *
	 * @return false
	 */
	virtual bool supports_binary() const { return false; }

	/** return whether likelihood function supports multiclass classification
	 *
	 * @return false
	 */
	virtual bool supports_multiclass() const { return false; }
};
}
#endif /* CLIKELIHOODMODEL_H_ */
