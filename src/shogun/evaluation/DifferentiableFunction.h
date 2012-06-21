/*
 * DifferentiableFunction.h
 *
 *  Created on: Jun 15, 2012
 *      Author: jacobw
 */

#ifndef DIFFERENTIABLEFUNCTION_H_
#define DIFFERENTIABLEFUNCTION_H_

#include <shogun/base/SGObject.h>
#include <shogun/lib/Map.h>
#include <shogun/lib/SGString.h>

namespace shogun {

class CDifferentiableFunction: public shogun::CSGObject {
public:
	CDifferentiableFunction();
	virtual ~CDifferentiableFunction();

	/** @return name of the SGSerializable */
	inline virtual const char* get_name() const	{ return "DifferentiableFunction"; }

	virtual CMap<SGString<char>, float64_t> get_gradient() = 0;
	virtual SGVector<float64_t> get_quantity() = 0;
};

} /* namespace shogun */
#endif /* DIFFERENTIABLEFUNCTION_H_ */
