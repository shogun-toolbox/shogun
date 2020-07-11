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

#ifndef ELASTICNETPENALTY_H
#define ELASTICNETPENALTY_H
#include <shogun/optimization/SparsePenalty.h>
#include <shogun/optimization/L1Penalty.h>
#include <shogun/optimization/L2Penalty.h>
namespace shogun
{
/** @brief The is the base class for ElasticNet penalty/regularization within the FirstOrderMinimizer framework.
 *
 * For ElasticNet penalty, \f$ElasticNet(w)\f$
 * \f[
 * ElasticNet(w)= \lambda \| w \|_1 + (1.0-\lambda) \| w \|_2
 * \f]
 * where \f$\lambda\f$ is the l1_ratio.
 *
 * Reference:
 * Zou, Hui, and Trevor Hastie. "Regularization and variable selection via the elastic net."
 * Journal of the Royal Statistical Society: Series B (Statistical Methodology) 67.2 (2005): 301-320.
 */

class ElasticNetPenalty: public SparsePenalty
{
public:
	ElasticNetPenalty()
		:SparsePenalty() {init();}

	~ElasticNetPenalty() override;

	/** returns the name of the class
	 *
	 * @return name ElasticNetPenalty
	 */
	const char* get_name() const override { return "ElasticNetPenalty"; }

	/** set l1_ratio
	 *
	 * @param ratio ratio must be in (0.0,1.0)
	 * */
	virtual void set_l1_ratio(float64_t ratio);

	/** Given the value of a target variable,
	 * this method returns the penalty of the variable 
	 *
	 * @param variable value of the variable
	 * @return penalty of the variable
	 */
	float64_t get_penalty(float64_t variable) override;

	/** Return the gradient of the penalty wrt a target variable
	 *
	 * @param variable value of a target variable
	 * @param gradient_of_variable unregularized/unpenalized gradient of the variable
	 * @return the gradient of the penalty wrt the variable
	 */
	float64_t get_penalty_gradient(float64_t variable,
		float64_t gradient_of_variable) override;

	/** Set the rounding epsilon for L1 penalty
	 *
	 * @param epsilon rounding epsilon
	 *
	 */
	virtual void set_rounding_epsilon(float64_t epsilon)
	{
		m_l1_penalty->set_rounding_epsilon(epsilon);
	}

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

	/** check l1_ratio */
	virtual void check_ratio();

	/** l1_ratio for L1 penalty and (1.0-l1_ratio) for L2 penalty */
	float64_t m_l1_ratio;

	/** L1Penalty */
	std::shared_ptr<L1Penalty> m_l1_penalty;

	/** L2Penalty */
	std::shared_ptr<L2Penalty> m_l2_penalty;

private:
	/**  init */
	void init();
};

}

#endif
