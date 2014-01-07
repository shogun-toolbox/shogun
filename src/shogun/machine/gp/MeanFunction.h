/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef CMEANFUNCTION_H_
#define CMEANFUNCTION_H_

#include <base/SGObject.h>
#include <base/Parameter.h>
#include <features/Features.h>

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
	 * @param features features to compute mean function
	 *
	 * @return mean of feature vectors
	 */
	virtual SGVector<float64_t> get_mean_vector(const CFeatures* features) const=0;

	/** returns the derivative of the mean function
	 *
	 * @param features features to compute mean function
	 * @param param parameter
	 * @param index of value if parameter is a vector
	 *
	 * @return derivative of mean function with respect to parameter
	 */
	virtual SGVector<float64_t> get_parameter_derivative(const CFeatures* features,
			const TParameter* param, index_t index=-1)
	{
		SG_ERROR("Can't compute derivative wrt %s parameter\n", param->m_name)
		return SGVector<float64_t>();
	}
};
}
#endif /* CMEANFUNCTION_H_ */
