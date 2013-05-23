/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 *
 * Code adapted from the GPML Toolbox:
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 *
 */

#ifndef CSTUDENTSTLIKELIHOOD_H_
#define CSTUDENTSTLIKELIHOOD_H_
#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3
#include <shogun/regression/gp/LikelihoodModel.h>

namespace shogun
{

/** @brief This is the class that models a likelihood model
 * with a Student's T Distribution. The parameters include
 * degrees of freedom as well as a sigma scale parameter.
 *
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

	/** destructor */
	virtual ~CStudentsTLikelihood();

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 * @return name of the SGSerializable
	 */
	virtual const char* get_name() const { return "StudentsTLikelihood"; }

	/** Returns the noise variance
	 *
	 * @return noise variance
	 */
	float64_t get_sigma() { return m_sigma; }

	/** Sets the noise variance
	 *
	 * @param sigma noise variance
	 */
	void set_sigma(float64_t sigma)
	{
		REQUIRE(sigma>0.0, "%s::set_sigma(): Standard deviation "
			"must be greater than zero\n", get_name())
		m_sigma=sigma;
	}

	/** get degrees of freedom
	 *
	 * @return degrees of freedom
	 */
	float64_t get_degrees_freedom() { return m_df; }

	/** sets degrees of freedom
	 *
	 * @param df degrees of freedom
	 */
	void set_degrees_freedom(float64_t df)
	{
		REQUIRE(df>1.0, "%s::set_degrees_freedom(): Number of degrees of "
				"freedom must be greater than one\n", get_name())
		m_df=df;
	}

	/** helper method used to specialize a base class instance
	 *
	 * @param likelihood likelihood model
	 * @return casted CStudentsTLikelihood object
	 */
	static CStudentsTLikelihood* obtain_from_generic(CLikelihoodModel* likelihood);

	/** Evaluate means
	 *
	 * @param means Vector of means calculated by inference method
	 * @return Final means evaluated by likelihood function
	 */
	virtual SGVector<float64_t> evaluate_means(SGVector<float64_t>& means);

	/** Evaluate variances
	 *
	 * @param vars Vector of variances calculated by inference method
	 * @return Final variances evaluated by likelihood function
	 */
	virtual SGVector<float64_t> evaluate_variances(SGVector<float64_t>& vars);

	/** get model type
	  *
	  * @return model type Student's T
	 */
	virtual ELikelihoodModelType get_model_type() { return LT_STUDENTST; }

	/** get log likelihood log(P(y|f)) with respect
	 *  to location f
	 *
	 *  @param labels labels used
	 *  @param f location
	 *
	 *  @return log likelihood
	 */
	virtual float64_t get_log_probability_f(CRegressionLabels* labels,
			SGVector<float64_t> f);

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
	virtual SGVector<float64_t> get_log_probability_derivative_f(
			CRegressionLabels* labels, SGVector<float64_t> f, index_t i);

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
	virtual SGVector<float64_t> get_first_derivative(CRegressionLabels* labels,
			TParameter* param, CSGObject* obj, SGVector<float64_t> function);

	/** get derivative of the first derivative
	 *  of log likelihood with respect to function
	 *  location, i.e.
	 *
	 *  \f$\frac{\partial}log(P(y|f))}{\partial{f}}\f$
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
	virtual SGVector<float64_t> get_second_derivative(CRegressionLabels* labels,
			TParameter* param, CSGObject* obj, SGVector<float64_t> function);

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
	virtual SGVector<float64_t> get_third_derivative(CRegressionLabels* labels,
			TParameter* param, CSGObject* obj, SGVector<float64_t> function);

private:
	/** Observation noise sigma */
	float64_t m_sigma;

	/** Degrees of Freedom */
	float64_t m_df;

	/** Initialize function */
	void init();
};
}
#endif /* HAVE_EIGEN3 */
#endif /* CStudentsTLIKELIHOOD_H_ */
