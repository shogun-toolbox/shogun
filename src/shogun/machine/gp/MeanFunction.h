/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef CMEANFUNCTION_H_
#define CMEANFUNCTION_H_

#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

/** @brief An abstract class of the mean function.
 *
 * This class takes the mean of data used for Gaussian Process Regression. It
 * also includes the derivatives of the specified function.
 */
class CMeanFunction : public CSGObject
{
public:
	/** constructor */
	CMeanFunction() { }

	virtual ~CMeanFunction() { }

	/** returns the mean of the specified data
	 *
	 * @param data points arranged in a matrix with rows representing the number
	 * of features
	 *
	 * @return mean of feature vectors
	 */
	virtual SGVector<float64_t> get_mean_vector(SGMatrix<float64_t> data) const=0;

	/** returns the derivative of the mean function
	 *
	 * @param data points arranged in a matrix with rows representing the number
	 * of features
	 * @param param parameter
	 * @param index of value if parameter is a vector
	 *
	 * @return derivative of mean function with respect to parameter
	 */
	virtual SGVector<float64_t> get_parameter_derivative(SGMatrix<float64_t> data,
			const TParameter* param, index_t index=-1)
	{
		SG_ERROR("Can't compute derivative wrt %s parameter\n", param->m_name)
		return SGVector<float64_t>();
	}
};
}
#endif /* CMEANFUNCTION_H_ */
