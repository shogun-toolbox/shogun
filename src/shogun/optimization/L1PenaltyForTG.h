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

#ifndef L1PENALTYFORTG_H
#define L1PENALTYFORTG_H
#include <shogun/optimization/L1Penalty.h>
#include <shogun/lib/SGVector.h>
namespace shogun
{
/** @brief The is the base class for L1 penalty/regularization within the FirstOrderMinimizer framework.
 *
 * For L1 penalty, \f$L1(w)\f$
 * \f[
 * L1(w)=\| w \|_1 = \sum_i \| w_i \|
 * \f]
 *
 * This class implements the truncated gradient method.
 *
 * Reference:
 * Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty
 */

class L1PenaltyForTG: public L1Penalty
{
public:
	/* Constructor */
	L1PenaltyForTG():L1Penalty() { init(); }

	/* Destructor */
	virtual ~L1PenaltyForTG() {}

	/** returns the name of the class
	 *
	 * @return name L1PenaltyForTG
	 */
	virtual const char* get_name() const { return "L1PenaltyForTG"; }

	/** Do proximal projection/operation in place
	 * @param variable the raw variable
	 * @param proximal_weight weight of the penalty
	 */
	virtual void update_variable_for_proximity(SGVector<float64_t> variable,
		float64_t proximal_weight);

protected:
	/** u is defined in Figure 2 of the reference */
	float64_t m_u;
	/** q is defined in Figure 2 of the reference */
	SGVector<float64_t> m_q;

private:
	/** init */
	void init();

};

}

#endif
