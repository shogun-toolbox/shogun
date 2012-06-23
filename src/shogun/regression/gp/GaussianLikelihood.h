/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef CGAUSSIANLIKELIHOOD_H_
#define CGAUSSIANLIKELIHOOD_H_

#include <shogun/regression/gp/LikelihoodModel.h>

namespace shogun {

/** @brief This is the class that models a Gaussian Likelihood
 *
 * In this case, P(y|f) is normally distributed with mean f and
 * variance $\sigma$.
 *
 */
class CGaussianLikelihood: public CLikelihoodModel {
  
public:

	/*Constructor*/
	CGaussianLikelihood();

	/*Destructor*/
	virtual ~CGaussianLikelihood();

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 * @return name of the SGSerializable
	 */
	inline virtual const char* get_name() const { return "GaussianLikelihood"; }

	/** Returns the noise variance
	 *
	 * @return noise variance
	 */
	float64_t get_sigma() {return m_sigma;}

	/** Sets the noise variance
	 *
	 * @param s noise variance
	 */
	void set_sigma(float64_t s) {m_sigma = s;}

	/** Evaluate means
	 *
	 * @param Vector of means calculated by inference method
	 * @return Final means evaluated by likelihood function
	 */
	virtual SGVector<float64_t> evaluate_means(SGVector<float64_t>& means);

	/** Evaluate variances
	 *
	 * @param Vector of variances calculated by inference method
	 * @return Final variances evaluated by likelihood function
	 */
	virtual SGVector<float64_t> evaluate_variances(SGVector<float64_t>& vars);
	
	/** get model type
	  *
	  * @return model type Gaussian
	 */
	virtual ELikelihoodModelType get_model_type() {return LT_GAUSSIAN;}

private:
	void init();

private:
	/** Observation noise sigma */
	float64_t m_sigma;
};

}

#endif /* CGAUSSIANLIKELIHOOD_H_ */
