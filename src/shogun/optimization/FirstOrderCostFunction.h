/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Wu Lin
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
 */

#ifndef FIRSTORDERCOSTFUNCTION_H
#define FIRSTORDERCOSTFUNCTION_H
#include <shogun/lib/config.h>
#include <shogun/lib/Map.h>
#include <shogun/lib/SGVector.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>
namespace shogun
{
/** @brief The first order cost function base class.
 *
 * This class gives the interface used in a first-order gradient-based unconstrained minimizer
 *
 */
class CFirstOrderCostFunction : public CSGObject
{
public:
	/** get the cost given current variables 
	 *
	 * @return cost
	 */
	virtual float64_t get_cost()=0;
	/** obtain reference of variables 
	 *
	 * @return reference of variables
	 */
	virtual CMap<TParameter*, SGVector<float64_t> >* obtain_variable_reference()=0;
	/** get the gradient value of variables 
	 *
	 * @return gradient of variables
	 */
	virtual CMap<TParameter*, SGVector<float64_t> >* get_gradient()=0;
};

}

#endif
