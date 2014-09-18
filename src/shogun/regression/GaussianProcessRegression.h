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

#ifdef HAVE_EIGEN3

#include <shogun/machine/GaussianProcessMachine.h>
#include <shogun/machine/gp/InferenceMethod.h>
#include <shogun/features/Features.h>
#include <shogun/labels/Labels.h>

namespace shogun
{

class CInferenceMethod;
class CFeatures;
class CLabels;

/** @brief Class GaussianProcessRegression implements regression based on
 * Gaussian Processes.
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
	CGaussianProcessRegression(CInferenceMethod* method);

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
	
	/** get covariance matrix
	 *
	 * @return covariance matrix
	 */
	SGMatrix<float64_t> get_covariance_matrix(CFeatures* data);
 
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
#endif /* HAVE_EIGEN3 */
#endif /* _GAUSSIANPROCESSREGRESSION_H_ */
