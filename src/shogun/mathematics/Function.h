/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#ifndef _FUNCTION_H_
#define _FUNCTION_H_

#include <base/SGObject.h>

namespace shogun
{

/** @brief Class of a function of one variable
 */
class CFunction : public CSGObject
{
public:
	/** returns the real value of the function at given point
	 *
	 * @param x argument of a function
	 *
	 * @return \f$f(x)\f$ - value of the function at given point
	 * \f$x\f$
	 */
	virtual float64_t operator() (float64_t x)=0;

	/** returns object name
	 *
	 * @return name Function
	 */
	virtual const char* get_name() const { return "Function"; }
};
}
#endif /* _FUNCTION_H_ */
