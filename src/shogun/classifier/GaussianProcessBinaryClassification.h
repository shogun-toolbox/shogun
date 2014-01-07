/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#ifndef _GAUSSIANPROCESSBINARYCLASSIFICATION_H_
#define _GAUSSIANPROCESSBINARYCLASSIFICATION_H_

#include <lib/config.h>

#ifdef HAVE_EIGEN3

#include <machine/GaussianProcessMachine.h>

namespace shogun
{

/** @brief Class GaussianProcessBinaryClassification implements binary
 * classification based on Gaussian Processes.
 */
class CGaussianProcessBinaryClassification : public CGaussianProcessMachine
{
public:
	/** problem type */
	MACHINE_PROBLEM_TYPE(PT_BINARY);

	/** default constructor */
	CGaussianProcessBinaryClassification();

	/** constructor
	 *
	 * @param method inference method
	 */
	CGaussianProcessBinaryClassification(CInferenceMethod* method);

	virtual ~CGaussianProcessBinaryClassification();

	/** apply machine to data in means of binary classification problem
	 *
	 * @param data (test) data to be classified
	 *
	 * @return classified labels
	 */
	virtual CBinaryLabels* apply_binary(CFeatures* data=NULL);

	/** returns a vector of of the posterior predictive means
	 *
	 * @param data (test) data to be classified
	 *
	 * @return mean vector
	 */
	SGVector<float64_t> get_mean_vector(CFeatures* data);

	/** returns a vector of the posterior predictive variances
	 *
	 * @param data (test) data to be classified
	 *
	 * @return variance vector
	 */
	SGVector<float64_t> get_variance_vector(CFeatures* data);

	/** returns probabilities \f$p(y_*=1)\f$ for each (test) feature \f$x_*\f$
	 *
	 * @param data (test) data to be classified
	 *
	 * @return vector of probabilities
	 */
	SGVector<float64_t> get_probabilities(CFeatures* data);

	/** get classifier type
	 *
	 * @return classifier type GAUSSIANPROCESSBINARY
	 */
	virtual EMachineType get_classifier_type()
	{
		return CT_GAUSSIANPROCESSBINARY;
	}

	/** return name of the classifier
	 *
	 * @return name GaussianProcessBinaryClassification
	 */
	virtual const char* get_name() const
	{
		return "GaussianProcessBinaryClassification";
	}

protected:
	/** train classifier
	 *
	 * @param data training data
	 *
	 * @return whether training was successful
	 */
	virtual bool train_machine(CFeatures* data=NULL);

	/** check whether training labels are valid for classification
	 *
	 * @param lab training labels
	 *
	 * @return whether training labels are valid for classification
	 */
	virtual bool is_label_valid(CLabels *lab) const
	{
		return (lab->get_label_type()==LT_BINARY);
	}
};
}
#endif /* HAVE_EIGEN3 */
#endif /* _GAUSSIANPROCESSCLASSIFICATION_H_ */
