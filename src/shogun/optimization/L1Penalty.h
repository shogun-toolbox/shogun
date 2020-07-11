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

#ifndef L1PENALTY_H
#define L1PENALTY_H
#include <shogun/optimization/SparsePenalty.h>

namespace shogun
{
/** @brief The is the base class for L1 penalty/regularization within the FirstOrderMinimizer framework.
 *
 * For L1 penalty, \f$L1(w)\f$
 * \f[
 * L1(w)=\| w \|_1 = \sum_i \| w_i \|
 * \f]
 *
 * This class implements the soft-threshold method.
 * Reference:
 * Proximal gradient method
 * www.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf
 */

class L1Penalty: public SparsePenalty
{
public:
	L1Penalty()
		:SparsePenalty() {init();}

	~L1Penalty() override {}

	/** Given the value of a target variable,
	 * this method returns the penalty of the variable 
	 *
	 * @param variable value of the variable
	 * @return penalty of the variable
	 */
	float64_t get_penalty(float64_t variable) override;

	/** returns the name of the class
	 *
	 * @return name L1Penalty
	 */
	const char* get_name() const override { return "L1Penalty"; }

	/** Return the gradient of the penalty wrt a target variable
	 *
	 * Note that for L1 penalty we do not compute the gradient/sub-gradient in our implementation.
	 * Instead, we do a proximal projection.
	 *
	 * @param variable value of a target variable
	 * @param gradient_of_variable unregularized/unpenalized gradient of the variable
	 * @return the gradient of the penalty wrt the variable
	 */

	float64_t get_penalty_gradient(float64_t variable,
		float64_t gradient_of_variable) override {return 0.0;}

	/** Set the rounding epsilon
	 *
	 * @param epsilon rounding epsilon
	 *
	 */
	virtual void set_rounding_epsilon(float64_t epsilon);

	/** Do proximal projection/operation in place
	 * @param variable the raw variable
	 * @param proximal_weight weight of the penalty
	 */
	void update_variable_for_proximity(SGVector<float64_t> variable,
		float64_t proximal_weight) override;

	/** Get the sparse variable
	 * @param variable the raw variable
	 * @param penalty_weight weight of the penalty
	 * @return sparse value of the variable
	 */
	float64_t get_sparse_variable(float64_t variable, float64_t penalty_weight) override;

protected:
	/** rounding epsilon */
	float64_t m_rounding_epsilon;

private:
	/** init */
	void init();
};

}

#endif
