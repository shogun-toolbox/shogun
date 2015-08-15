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
namespace shogun
{
/** @brief The base class for penalty/regularization used in minimization.
 *
 * This is the interface of regularizers used in minimizers.
 * (eg,
 * FirstOrderMinimizer::update_gradient(SGVector<float64_t> gradient, SGVector<float64_t> var)
 * FirstOrderMinimizer::get_penalty(SGVector<float64_t> var)
 * ) 
 *
 */
class Penalty
{
public:
	/** Given the value of a target variable,
	 * this method returns the penalty of the variable 
	 *
	 * For L2 penalty, \f$L2(w)\f$
	 * \f[
	 * L2(w)=\frac{w^t w}{2}
	 * \f]
	 *
	 * @param variable value of the variable
	 * @return penalty of the variable
	 */
	virtual float64_t get_penalty(float64_t variable)=0;

	/** Returns the gradient of the penalty wrt a target variable
	 * Note that the penalized gradient=unpenalized gradient+penalty_gradient
	 *
	 * For L2 penalty
	 * \f[
	 * \frac{\partial L2(w) }{\partial w}=w
	 * \f]
	 *
	 * @param variable value of a target variable
	 * @param gradient unregularized/unpenalized gradient of the variable
	 * @return the gradient of the penalty wrt the variable
	 */
	virtual float64_t get_penalty_gradient(float64_t variable,
		float64_t gradient)=0;
};

}
#endif
