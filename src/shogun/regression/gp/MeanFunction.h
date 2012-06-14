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
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

namespace shogun {

/** @brief Mean Function base class.
 *
 *  The Mean Function Class takes the mean
 *  of data used for Gaussian Process Regression.
 *  It also includes the derivatives of the
 *  specified function.
 *
 */
class CMeanFunction: public shogun::CSGObject {

public:

	/*Constructor*/
	CMeanFunction();

	/*Destructor*/
	virtual ~CMeanFunction();

	/** Returns the mean of the specified data
	 *
	 * @param data points arranged in a matrix with rows representing the number of features
	 *
	 * @return Mean of feature vectors
	 */
	virtual SGVector<float64_t> get_mean_vector(SGMatrix<float64_t>& data) = 0;

	/** Returns the mean of the specified data
	 *
	 * @param data points arranged in a matrix with rows representing the number of features
	 *
	 * @param param_name Name of parameters
	 *
	 * @return derivative of mean function with respect to parameter
	 */
	virtual float64_t get_parameter_derivative(SGMatrix<float64_t>& data, const char* param_name);

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 * @return name of the SGSerializable
	 */
	inline virtual const char* get_name() const { return "MeanFunction"; }

};

}

#endif /* CMEANFUNCTION_H_ */
