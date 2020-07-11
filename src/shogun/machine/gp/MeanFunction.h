/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2013 Roman Votyakov
 * Written (W) 2012 Jacob Walker
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
 */
#ifndef CMEANFUNCTION_H_
#define CMEANFUNCTION_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/features/Features.h>

namespace shogun
{

/** @brief An abstract class of the mean function.
 *
 * This class takes the mean of data used for Gaussian Process Regression. It
 * also includes the derivatives of the specified function.
 */
class MeanFunction : public SGObject
{
public:
	/** constructor */
	MeanFunction() { }

	~MeanFunction() override { }

	/** returns the mean of the specified data
	 *
	 * @param features features to compute mean function
	 *
	 * @return mean of feature vectors
	 */
	virtual SGVector<float64_t> get_mean_vector(std::shared_ptr<const Features> features) const=0;

#ifndef SWIG
	/** returns the derivative of the mean function
	 *
	 * @param features features to compute mean function
	 * @param param parameter
	 * @param index of value if parameter is a vector
	 *
	 * @return derivative of mean function with respect to parameter
	 */
	virtual SGVector<float64_t> get_parameter_derivative(std::shared_ptr<const Features> features,
			Parameters::const_reference param, index_t index=-1)
	{
		error("Can't compute derivative wrt {} parameter", param.first);
		return SGVector<float64_t>();
	}
#endif
};
}
#endif /* CMEANFUNCTION_H_ */
