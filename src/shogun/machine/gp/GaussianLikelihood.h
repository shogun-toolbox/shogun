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
 * The hyperparameter of the Gaussian likelihood model is standard
 * deviation: \f$\sigma\f$.
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
	 * @param likelihood likelihood model
	 * @return casted CGaussianLikelihood object
	 */
	static CGaussianLikelihood* obtain_from_generic(CLikelihoodModel* likelihood);

	/** evaluate means
	 *
	 * @param means vector of means calculated by inference method
	 * @return final means evaluated by likelihood function
	 */
	virtual SGVector<float64_t> evaluate_means(SGVector<float64_t>& means);

	/** evaluate variances
	 *
	 * @param vars vector of variances calculated by inference method
	 * @return final variances evaluated by likelihood function
	 */
	virtual SGVector<float64_t> evaluate_variances(SGVector<float64_t>& vars);

	/** get model type
	 *
	 * @return model type Gaussian
	 */
	virtual ELikelihoodModelType get_model_type() { return LT_GAUSSIAN; }

	/** get log likelihood \f$log(P(y|f))\f$
	 *
	 * @param labels labels used
	 * @param f function location
	 *
	 * @return log likelihood
	 */
	virtual float64_t get_log_probability_f(CRegressionLabels* labels,
			SGVector<float64_t> f);

	/** get derivative of log likelihood \f$log(P(y|f))\f$ with
	 * respect to function location \f$f\f$
	 *
	 * @param labels labels used
	 * @param f function location
	 * @param i index, choices are 1, 2, and 3 for first, second, and
	 * third derivatives respectively
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_log_probability_derivative_f(
			CRegressionLabels* labels, SGVector<float64_t> f, index_t i);

	/** get derivative of log likelihood \f$log(P(y|f))\f$ with
	 * respect to given parameter
	 *
	 * @param labels labels used
	 * @param param parameter
	 * @param obj pointer to object to make sure we have the right
	 * parameter
	 * @param function function location
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_first_derivative(CRegressionLabels* labels,
			TParameter* param, CSGObject* obj, SGVector<float64_t> function);

	/** get derivative of the first derivative of log likelihood with
	 * respect to function location, i.e. \f$\frac{\partial
	 * log(P(y|f))}{\partial f}\f$ with respect to given parameter
	 *
	 * @param labels labels used
	 * @param param parameter
	 * @param obj pointer to object to make sure we have the right
	 * parameter
	 * @param function function location
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_second_derivative(CRegressionLabels* labels,
			TParameter* param, CSGObject* obj, SGVector<float64_t> function);

	/** get derivative of the second derivative of log likelihood with
	 * respect to function location, i.e. \f$\frac{\partial^{2}
	 * log(P(y|f))}{\partial f^{2}}\f$ with respect to given
	 * parameter
	 *
	 * @param labels labels used
	 * @param param parameter
	 * @param obj pointer to object to make sure we have the right
	 * parameter
	 * @param function function location
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_third_derivative(CRegressionLabels* labels,
			TParameter* param, CSGObject* obj, SGVector<float64_t> function);

	/** return whether Gaussian likelihood function supports regression
	 *
	 * @return true
	 */
	virtual bool supports_regression() { return true; }

private:
	/** initialize function */
	void init();

	/** standard deviation */
	float64_t m_sigma;
};
}
#endif /* HAVE_EIGEN3 */
#endif /* CGAUSSIANLIKELIHOOD_H_ */
