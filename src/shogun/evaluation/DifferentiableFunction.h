/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef CDIFFERENTIABLEFUNCTION_H_
#define CDIFFERENTIABLEFUNCTION_H_

#include <shogun/base/SGObject.h>
#include <shogun/lib/Map.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

/** @brief An abstract class that describes a differentiable function used for
 * GradientEvaluation.
 */
class CDifferentiableFunction : public CSGObject
{
public:
	/** default constructor */
	CDifferentiableFunction();

	virtual ~CDifferentiableFunction();

	/** get the gradient
	 *
	 * @param parameters parameter's dictionary
	 *
	 * @return map of gradient. Keys are names of parameters, values are values
	 * of derivative with respect to that parameter.
	 */
	virtual CMap<TParameter*, SGVector<float64_t> >* get_gradient(
			CMap<TParameter*, CSGObject*>* parameters)=0;

	/** get the function value
	 *
	 * @return vector that represents the function value
	 */
	virtual SGVector<float64_t> get_value()=0;
};
}
#endif /* CDIFFERENTIABLEFUNCTION_H_ */
