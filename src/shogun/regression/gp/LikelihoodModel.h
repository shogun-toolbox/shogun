/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef CLIKELIHOODMODEL_H_
#define CLIKELIHOODMODEL_H_

#include <shogun/base/SGObject.h>

namespace shogun {

/** @brief The Likelihood Model base class.
 *
 *  The Likelihood model computes approximately the
 *  distribution P(y|f), where y are the labels, and f
 *  is the prediction function.
 *
 */
class CLikelihoodModel : public CSGObject {
public:
	/*Constructor*/
	CLikelihoodModel();
	virtual ~CLikelihoodModel();

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 * @return name of the SGSerializable
	 */
	inline virtual const char* get_name() const { return "LikelihoodModel"; }

	/** get likelihood function derivative with respect to parameters
	 *
	 * @param name of parameter used to take derivative
	 * @return likelihood derivative with respect to parameter
	 */
	virtual float64_t get_parameter_derivative(const char* param_name);
};
}

#endif /* CLIKELIHOODMODEL_H_ */
