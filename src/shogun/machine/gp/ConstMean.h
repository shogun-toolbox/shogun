/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
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
 * Adapted from the GPML toolbox, specifically meanConst.m
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 */

#ifndef CCONSTMEAN_H_
#define CCONSTMEAN_H_

#include <shogun/lib/config.h>

#include <shogun/machine/gp/MeanFunction.h>

namespace shogun
{

/** @brief The Const mean function class.
 *
 * Simple mean function that assumes a mean of Const value.
 */
class CConstMean : public CMeanFunction
{
public:
	/** default constructor
	 * the default value of mean is 0
	 */
	CConstMean();

	/** constructor
	 * @param mean const value for mean function
	 */
	CConstMean(float64_t mean);

	virtual ~CConstMean();

	/** set the const_value of mean function
	 *
	 * @param mean const value for mean function
	 */
	virtual void set_const(float64_t mean) {m_mean=mean;}

	/** returns name of the mean function
	 *
	 * @return name ConstMean
	 */
	virtual const char* get_name() const { return "ConstMean"; }

	/** returns the mean of the specified data
	 *
	 * @param features data points arranged in a matrix with rows representing the number
	 * of features
	 *
	 * @return mean of feature vectors
	 */
	virtual SGVector<float64_t> get_mean_vector(const CFeatures* features) const;

	/** returns the derivative of the mean function
	 *
	 * @param features features to compute mean function
	 * @param param parameter
	 * @param index of value if parameter is a vector
	 *
	 * @return derivative of mean function with respect to parameter
	 */
	virtual SGVector<float64_t> get_parameter_derivative(const CFeatures* features,
			const TParameter* param, index_t index=-1);
private:

	void init();

	/** const value of mean function*/
	float64_t m_mean;
};
}
#endif /* CCONSTMEAN_H_ */
