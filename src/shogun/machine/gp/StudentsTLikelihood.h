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
 * p(y|f)=\prod_{i=1}^{n} \frac{\Gamma(\frac{\nu+1}{2})} {\Gamma(\frac{\nu}{2})\sqrt{\nu\pi}\sigma}
 * \left(1+\frac{(y_i-f_i)^2}{\nu\sigma^2} \right)^{-\frac{\nu+1}{2}}
 * \f]
 *
 * The hyperparameters of the Student's t-likelihood model are
 * \f$\sigma\f$ - scale parameter, and \f$\nu\f$ - degrees of freedom.
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
	 * @return model type Student's-t
	 */
	virtual ELikelihoodModelType get_model_type() { return LT_STUDENTST; }

	/** get log likelihood \f$log(P(y|f))\f$
	 *
	 * @param lab labels used
	 * @param func function location
	 *
	 * @return log likelihood
	 */
	virtual float64_t get_log_probability_f(CLabels* lab,
			SGVector<float64_t> func);

	/** get derivative of log likelihood \f$log(P(y|f))\f$ with
	 * respect to function location \f$f\f$
	 *
	 * @param lab labels used
	 * @param func function location
	 * @param i index, choices are 1, 2, and 3 for first, second, and
	 * third derivatives respectively
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_log_probability_derivative_f(
			CLabels* lab, SGVector<float64_t> func, index_t i);

	/** get derivative of log likelihood \f$log(P(y|f))\f$ with
	 * respect to given parameter
	 *
	 * @param lab labels used
	 * @param param parameter
	 * @param obj pointer to object to make sure we have the right
	 * parameter
	 * @param func function location
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_first_derivative(CLabels* lab,
			TParameter* param, CSGObject* obj, SGVector<float64_t> func);

	/** get derivative of the first derivative of log likelihood with
	 * respect to function location, i.e. \f$\frac{\partial
	 * log(P(y|f))}{\partial f}\f$ with respect to given parameter
	 *
	 * @param lab labels used
	 * @param param parameter
	 * @param obj pointer to object to make sure we have the right
	 * parameter
	 * @param func function location
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_second_derivative(CLabels* lab,
			TParameter* param, CSGObject* obj, SGVector<float64_t> func);

	/** get derivative of the second derivative of log likelihood with
	 * respect to function location, i.e. \f$\frac{\partial^{2}
	 * log(P(y|f))}{\partial f^{2}}\f$ with respect to given
	 * parameter
	 *
	 * @param lab labels used
	 * @param param parameter
	 * @param obj pointer to object to make sure we have the right
	 * parameter
	 * @param func function location
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_third_derivative(CLabels* lab,
			TParameter* param, CSGObject* obj, SGVector<float64_t> func);

	/** return whether Student's likelihood function supports
	 * regression
	 *
	 * @return true
	 */
	virtual bool supports_regression() { return true; }

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
