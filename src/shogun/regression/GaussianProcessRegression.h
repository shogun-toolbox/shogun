/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jacob Walker, Sergey Lisitsyn, Heiko Strathmann, Roman Votyakov, 
 *          Soeren Sonnenburg, Wu Lin, Fernando Iglesias
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

class Inference;
class Features;
class Labels;

/** @brief Class GaussianProcessRegression implements regression based on
 * Gaussian Processes.
 */
class GaussianProcessRegression : public GaussianProcessMachine
{
public:
	/** problem type */
	MACHINE_PROBLEM_TYPE(PT_REGRESSION);

	/** default constructor */
	GaussianProcessRegression();

	/** constructor
	 *
	 * @param method chosen inference method
	 */
	GaussianProcessRegression(const std::shared_ptr<Inference>& method);

	~GaussianProcessRegression() override;

	/** apply regression to data
	 *
	 * @param data (test)data to be classified
	 * @return classified labels
	 */
	std::shared_ptr<RegressionLabels> apply_regression(std::shared_ptr<Features> data=NULL) override;

	/** get predicted mean vector
	 *
	 * @return predicted mean vector
	 */
	SGVector<float64_t> get_mean_vector(const std::shared_ptr<Features>& data);

	/** get variance vector
	 *
	 * @return variance vector
	 */
	SGVector<float64_t> get_variance_vector(const std::shared_ptr<Features>& data);

	/** get classifier type
	 *
	 * @return classifier type GaussianProcessRegression
	 */
	EMachineType get_classifier_type() override
	{
		return CT_GAUSSIANPROCESSREGRESSION;
	}

	/** return name of the regression object
	 *
	 * @return name GaussianProcessRegression
	 */
	const char* get_name() const override { return "GaussianProcessRegression"; }

protected:
	/** train regression
	 *
	 * @param data training data
	 *
	 * @return whether training was successful
	 */
	bool train_machine(std::shared_ptr<Features> data=NULL) override;

	/** check whether training labels are valid for regression
	 *
	 * @param lab training labels
	 *
	 * @return whether training labels are valid for regression
	 */
	bool is_label_valid(std::shared_ptr<Labels >lab) const override
	{
		return lab->get_label_type()==LT_REGRESSION;
	}
};
}
#endif /* _GAUSSIANPROCESSREGRESSION_H_ */
