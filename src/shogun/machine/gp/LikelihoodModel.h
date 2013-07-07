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

#ifndef CLIKELIHOODMODEL_H_
#define CLIKELIHOODMODEL_H_

#include <shogun/base/SGObject.h>
#include <shogun/labels/Labels.h>

namespace shogun
{

/** type of likelihood model */
enum ELikelihoodModelType
{
	LT_NONE = 0,
	LT_GAUSSIAN = 10,
	LT_STUDENTST = 20,
	LT_LOGIT = 30,
	LT_PROBIT = 40
};

/** @brief The Likelihood model base class.
 *
 * The Likelihood model computes approximately the distribution
 * \f$p(y|f)\f$, where \f$y\f$ are the labels, and \f$f\f$ is the
 * prediction function.
 */
class CLikelihoodModel : public CSGObject
{
public:
	/** constructor */
	CLikelihoodModel();

	virtual ~CLikelihoodModel();

	/** evaluate means
	 *
	 * @param mu vector of means calculated by inference method
	 * @param s2 vector of variances calculated by inference method
	 * @return final means evaluated by likelihood function
	 */
	virtual SGVector<float64_t> evaluate_means(SGVector<float64_t> mu,
			SGVector<float64_t> s2)=0;

	/** evaluate variances
	 *
	 * @param mu vector of means calculated by inference method
	 * @param s2 vector of variances calculated by inference method
	 * @return final variances evaluated by likelihood function
	 */
	virtual SGVector<float64_t> evaluate_variances(SGVector<float64_t> mu,
			SGVector<float64_t> s2)=0;

	/** get model type
	  *
	  * @return model type NONE
	 */
	virtual ELikelihoodModelType get_model_type() { return LT_NONE; }

	/** get log likelihood \f$log(P(y|f))\f$
	 *
	 * @param lab labels used
	 * @param func function location
	 *
	 * @return log likelihood
	 */
	virtual SGVector<float64_t> get_log_probability_f(CLabels* lab,
			SGVector<float64_t> func)=0;

	/** get derivative of log likelihood \f$log(P(y|f))\f$ with
	 * respect to location function \f$f\f$
	 *
	 * @param lab labels used
	 * @param func function location
	 * @param i index, choices are 1, 2, and 3 for first, second, and
	 * third derivatives respectively
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_log_probability_derivative_f(
			CLabels* lab, SGVector<float64_t> func, index_t i)=0;

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
			TParameter* param, CSGObject* obj, SGVector<float64_t> func)=0;

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
			TParameter* param, CSGObject* obj, SGVector<float64_t> func)=0;

	/** get derivative of the second derivative of log likelihood with
	 * respect to function location, i.e. \f$\frac{\partial^{2}
	 * log(P(y|f))}{\partial f^{2}}\f$ with respect to given parameter
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
			TParameter* param, CSGObject* obj, SGVector<float64_t> func)=0;

	/** return whether likelihood function supports regression
	 *
	 * @return false
	 */
	virtual bool supports_regression() { return false; }

	/** return whether likelihood function supports binary
	 * classification
	 *
	 * @return false
	 */
	virtual bool supports_binary() { return false; }

	/** return whether likelihood function supports multiclass
	 * classification
	 *
	 * @return false
	 */
	virtual bool supports_multiclass() { return false; }
};
}
#endif /* CLIKELIHOODMODEL_H_ */
