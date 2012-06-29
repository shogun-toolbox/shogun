/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef CDIFFERENTIABLEFUNCTION_H_
#define CDIFFERENTIABLEFUNCTION_H_

#include <shogun/base/SGObject.h>
#include <shogun/lib/Map.h>
#include <shogun/lib/SGString.h>

namespace shogun
{

/** @brief DifferentiableFunction.
 *
 * This is an interface that describes a differentiable function
 * used for GradientEvaluation.
 *
 */
class CDifferentiableFunction: public CSGObject
{

public:

	/*Constructor*/
	CDifferentiableFunction();

	/*Destructor*/
	virtual ~CDifferentiableFunction();

	/*Get the gradient
	 *
	 * @return Map of gradient. Keys are names of parameters, values are
	 * values of derivative with respect to that parameter.
	 */
	virtual CMap<SGString<const char>, float64_t> get_gradient() = 0;

	/*Get the function value
	 *
	 * @return Vector that represents the function value
	 */
	virtual SGVector<float64_t> get_quantity() = 0;
};

} /* namespace shogun */

#endif /* CDIFFERENTIABLEFUNCTION_H_ */
