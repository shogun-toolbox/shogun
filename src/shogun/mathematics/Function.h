/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Roman Votyakov, Yuyu Zhang
 */

#ifndef _FUNCTION_H_
#define _FUNCTION_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>

namespace shogun
{

/** @brief Class of a function of one variable
 */
class Function : public SGObject
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
