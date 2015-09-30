/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
 * Written (W) 2013 Roman Votyakov
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 * Code adapted from
 * Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * and 
 * https://gist.github.com/yorkerlin/8a36e8f9b298aa0246a4
 */

#ifndef _GAUSSIANPROCESSCLASSIFICATION_H_
#define _GAUSSIANPROCESSCLASSIFICATION_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/GaussianProcessMachine.h>
#include <shogun/machine/Machine.h>

namespace shogun
{

/** @brief Class GaussianProcessClassification implements binary and multiclass
 * classification based on Gaussian Processes.
 */
class CGaussianProcessClassification : public CGaussianProcessMachine
{
public:
	/** problem type */
	MACHINE_PROBLEM_TYPE(PT_CLASS);

	/** default constructor */
	CGaussianProcessClassification();

	/** constructor
	 *
	 * @param method inference method
	 */
	CGaussianProcessClassification(CInferenceMethod* method);

	virtual ~CGaussianProcessClassification();

	/** apply machine to data in means of binary classification problem
	 *
	 * @param data (test) data to be classified
	 *
	 * @return classified labels (label is either -1 or 1)
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
	 * @return classifier type GAUSSIANPROCESS
	 */
	virtual EMachineType get_classifier_type()
	{
		return CT_GAUSSIANPROCESSCLASS;
	}

	/** return name of the classifier
	 *
	 * @return name GaussianProcessClassification
	 */
	virtual const char* get_name() const
	{
		return "GaussianProcessClassification";
	}
	/** apply machine to data in means of multi class classification problem
	 *
	 * @param data (test) data to be classified
	 *
	 * @return classified labels (label starts from 0)
	 */
	virtual CMulticlassLabels* apply_multiclass(CFeatures* data=NULL);

protected:
	/** train classifier
	 *
	 * @param data training data
	 *
	 * @return whether training was successful
	 */
	virtual bool train_machine(CFeatures* data=NULL);

};
}
#endif /* HAVE_EIGEN3 */
#endif /* _GAUSSIANPROCESSCLASSIFICATION_H_ */
