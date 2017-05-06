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

#ifndef _GAUSSIANPROCESSREGRESSION_H_
#define _GAUSSIANPROCESSREGRESSION_H_


#include <shogun/lib/config.h>
#include <shogun/machine/GaussianProcessMachine.h>
#include <shogun/machine/gp/Inference.h>
#include <shogun/features/Features.h>
#include <shogun/labels/Labels.h>

namespace shogun
{

class CInference;
class CFeatures;
class CLabels;

/** @brief  
 * Description: Class GaussianProcessRegression implements a Gaussian Process Regression(GPR). It is a nonparametric kernel-based probabilistic regressor. 
 *              Building onto a linear regression model, y=x^{T}\cdot \boldsymbol{\beta }+\epsilon, it adds
 *              in latent variables, f(x)_{i},i=1,2,...,n, from a Gaussian Process and a kernel function, which becomes 
 *              y=h(x)^{T}\cdot \boldsymbol{\beta }+f(x), f(x) is from a zero mean Gaussian Processor with covariance function, k(x,x').
 *              Plotting regression error is viable for GPR. This regressor is implemented was based on Matlab's GPR.
 * Computation load: computational complexity: O(n^3); Memory: O(n^2)
 * Reference: 
 * 1. Wikipedia page for Gaussian_process: https://en.wikipedia.org/wiki/Gaussian_process
 * 2. http://www.gaussianprocess.org/gpml/chapters/RW2.pdf
 * [C. E. Rasmussen and C. K. I. Williams. Gaussian Processes for Machine Learning (Adaptive Computation and Machine Learning). The MIT Press, 2005.]
 * 
 */
	
class CGaussianProcessRegression : public CGaussianProcessMachine
{
public:
	/** problem type */
	MACHINE_PROBLEM_TYPE(PT_REGRESSION);

	/** default constructor */
	CGaussianProcessRegression();

	/** constructor
	 *
	 * @param method chosen inference method
	 */
	CGaussianProcessRegression(CInference* method);

	virtual ~CGaussianProcessRegression();

	/** apply regression to data
	 *
	 * @param data (test)data to be classified
	 * @return classified labels
	 */
	virtual CRegressionLabels* apply_regression(CFeatures* data=NULL);

	/** get predicted mean vector
	 *
	 * @return predicted mean vector
	 */
	SGVector<float64_t> get_mean_vector(CFeatures* data);

	/** get variance vector
	 *
	 * @return variance vector
	 */
	SGVector<float64_t> get_variance_vector(CFeatures* data);

	/** get classifier type
	 *
	 * @return classifier type GaussianProcessRegression
	 */
	virtual EMachineType get_classifier_type()
	{
		return CT_GAUSSIANPROCESSREGRESSION;
	}

	/** return name of the regression object
	 *
	 * @return name GaussianProcessRegression
	 */
	virtual const char* get_name() const { return "GaussianProcessRegression"; }

protected:
	/** train regression
	 *
	 * @param data training data
	 *
	 * @return whether training was successful
	 */
	virtual bool train_machine(CFeatures* data=NULL);

	/** check whether training labels are valid for regression
	 *
	 * @param lab training labels
	 *
	 * @return whether training labels are valid for regression
	 */
	virtual bool is_label_valid(CLabels *lab) const
	{
		return lab->get_label_type()==LT_REGRESSION;
	}
};
}
#endif /* _GAUSSIANPROCESSREGRESSION_H_ */
