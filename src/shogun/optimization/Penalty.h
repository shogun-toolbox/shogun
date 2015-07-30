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

#ifndef PENALTY_H
#define PENALTY_H
#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
namespace shogun
{
/** @brief The base class for penalty/regularization used in minimization.
 *
 * this is an interface used in minimization
 *
 */
class CPenalty
{
public:
	/** Given the value of a variable, this method returns the penalty of this variable 
	 *
	 * For example, for L2 penalty
	 * penalty=0.5*variable*variable
	 *
	 * @param variable value of the variable
	 * @return penalty of the variable
	 */
	virtual float64_t get_penalty(float64_t variable)=0;

	/** this method returns the gradient with respect to the penalty
	 *
	 * For example, for L2 penalty
	 * gradient_wrt_penalty=variable
	 * Note that penalized gradient=unpenalized gradient+gradient_wrt_penalty
	 *
	 * @param variable value of the variable
	 * @param gradient unregularized/unpenalized gradient of the variable
	 * @return the gradient of penalty of the variable
	 */
	virtual float64_t get_gradient_wrt_penalty(float64_t variable,
		float64_t gradient_of_variable)=0;

};

}
#endif
