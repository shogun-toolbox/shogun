/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef CLIKELIHOODMODEL_H_
#define CLIKELIHOODMODEL_H_

#include <shogun/base/SGObject.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/labels/RegressionLabels.h>


namespace shogun
{
  
/** Type of likelihood model*/
enum ELikelihoodModelType
{
	LT_NONE = 0,
	LT_GAUSSIAN = 10,
	LT_STUDENTST = 20
};
	

/** @brief The Likelihood Model base class.
 *
 *  The Likelihood model computes approximately the
 *  distribution P(y|f), where y are the labels, and f
 *  is the prediction function.
 *
 */
class CLikelihoodModel : public CSGObject
{

public:
  
	/*Constructor*/
	CLikelihoodModel();

	/*Destructor*/
	virtual ~CLikelihoodModel();

	/** get likelihood function derivative with respect to parameters
	 *
	 * @param param_name name of parameter used to take derivative
	 * @return likelihood derivative with respect to parameter
	 */
	virtual float64_t get_parameter_derivative(const char* param_name);

	/** Evaluate means
	 *
	 * @param means Vector of means calculated by inference method
	 * @return Final means evaluated by likelihood function
	 */
	virtual SGVector<float64_t> evaluate_means(SGVector<float64_t>& means) = 0;

	/** Evaluate variances
	 *
	 * @param vars Vector of variances calculated by inference method
	 * @return Final variances evaluated by likelihood function
	 */
	virtual SGVector<float64_t> evaluate_variances(SGVector<float64_t>& vars) = 0;
	
	/** get model type
	  *
	  * @return model type NONE
	 */
	virtual ELikelihoodModelType get_model_type() {return LT_NONE;}

	/** get degrees of freedom (1 if
	 * degrees of freedom not supported
	 * by likelihood function)
	 *
	 * @return degrees of freedom
	 */
	inline virtual float64_t get_degrees_freedom() {return m_df;}

	/** get log likelihood log(P(y|f)) with respect
	 *  to location f
	 *
	 *  @param labels labels used
	 *  @param f location
	 *
	 *  @return log likelihood
	 */
	virtual float64_t get_log_probability_f(CRegressionLabels* labels,
			Eigen::VectorXd f) = 0;


	/** get derivative of log likelihood log(P(y|f)) with respect
	 *  to location f
	 *
	 *  @param labels labels used
	 *  @param f location
	 *  @param i index, choices are 1, 2, and 3
	 *  for first, second, and third derivatives
	 *  respectively
	 *
	 *  @return derivative
	 */
	virtual Eigen::VectorXd get_log_probability_derivative_f(
			CRegressionLabels* labels, Eigen::VectorXd f, index_t i) = 0;

	/** get derivative of log likelihood log(P(y|f))
	 *  with respect to given parameter
	 *
	 *  @param labels labels used
	 *  @param param parameter
	 *  @param obj pointer to object to make sure we
	 *  have the right parameter
	 *  @param function function location
	 *
	 *  @return derivative
	 */
	virtual Eigen::VectorXd get_first_derivative(CRegressionLabels* labels,
			TParameter* param, CSGObject* obj, Eigen::VectorXd function) = 0;

	/** get derivative of the second derivative
	 *  of log likelihood with respect to function
	 *  location, i.e.
	 *
	 *  \f$\frac{\partial^{2}log(P(y|f))}{\partial{f^{2}}}\f$
	 *
	 *  with respect to given parameter
	 *
	 *  @param labels labels used
	 *  @param param parameter
	 *  @param obj pointer to object to make sure we
	 *  have the right parameter
	 *  @param function function location
	 *
	 *  @return derivative
	 */
	virtual Eigen::VectorXd get_second_derivative(CRegressionLabels* labels,
			TParameter* param, CSGObject* obj, Eigen::VectorXd function) = 0;
protected:

	/** Degrees of Freedom*/
	float64_t m_df;

};


}

#endif /* CLIKELIHOODMODEL_H_ */
