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

#ifndef SVRGMINIMIZER_H
#define SVRGMINIMIZER_H
#include <shogun/optimization/FirstOrderStochasticMinimizer.h>
#include <shogun/optimization/FirstOrderSAGCostFunction.h>
namespace shogun
{

/** @brief The class implements the stochastic variance reduced gradient (SVRG) minimizer.
 *
 * Reference:
 * Johnson, Rie, and Tong Zhang.
 * "Accelerating stochastic gradient descent using predictive variance reduction."
 * Advances in Neural Information Processing Systems. 2013.
 */

class SVRGMinimizer: public FirstOrderStochasticMinimizer
{
public:
	/** Default constructor */
	SVRGMinimizer();

	/** Constructor
	 * @param fun stochastic cost function
	 */
	SVRGMinimizer(FirstOrderSAGCostFunction *fun);

	/** Destructor */
	virtual ~SVRGMinimizer();

	/** Do minimization and get the optimal value 
	 * 
	 * @return optimal value
	 */
	virtual float64_t minimize();

	/** Set the number to go through data using SGDMinimizer to initialize variables before SVRG minimization
	 *
	 * @param sgd_passes the number to go through data using SGDMinimizer
	 */

	virtual void set_sgd_number_passes(int32_t sgd_passes)
	{
		REQUIRE(m_num_passes>0, "Must set num_passes first\n");
		REQUIRE(sgd_passes>=0, "The number (%d) to go through data using SGD update must be positive\n",
			sgd_passes);
		REQUIRE(m_num_passes>sgd_passes,
			"SVRG update is not actived because the total number (%d) to go through data",
			" is less than the number (%d) to go through data using SGD\n",
			m_num_passes, sgd_passes);
		m_num_sgd_passes=sgd_passes;
	}

	/** Set the number of interval to average stochastic sample gradients 
	 *
	 * If we have \f$(n-g)\f$ passes to go through data and the interval is \f$k\f$, we will average stochastic sample
	 * gradients at the 0-th, k-th, 2k-th, 3k-th, ... pass
	 *
	 * Note that \f$n\f$ is the total number to go through data and \f$g\f$ is the number of
	 * using SGDMinimizer to initialize variables, 
	 *
	 * @param interval how often to average stochastic sample gradients
	 */
	virtual void set_average_update_interval(int32_t interval)
	{
		REQUIRE(m_num_passes>0, "Must set num passes first\n");
		REQUIRE(m_num_sgd_passes>=0, "Must set sgd update passes first\n");
		REQUIRE(interval>0, "Interval (%d) must be positive\n", interval);
		REQUIRE((m_num_passes-m_num_sgd_passes)%interval==0, "Interval is not valid\n");
		/* if (m_num_passes-m_num_sgd_passes)%interval!=0, will affect the finaly result if we do the following operations:
		* first do minimization, then save_to_context, and then load_from_context and finaly do minimization.
		* If we want to get the exact result when (m_num_passes-m_num_sgd_passes)%interval!=0,
		* we should store/restore m_average_gradient and m_previous_variable in save_to_context/load_from_context
		*/
		m_svrg_interval=interval;
	}

protected:
	/*  init the minimization process */
	virtual void init_minimization();

	/* the number to go through data  using SGD before SVRG update */
	int32_t m_num_sgd_passes;

	/* interval to average gradient */
	int32_t m_svrg_interval;

	/*  used to store average gradient */
	SGVector<float64_t> m_average_gradient;

	/*  used to store previous result */
	SGVector<float64_t> m_previous_variable;
private:
	/* Init */
	void init();
};

}
#endif /* SVRGMINIMIZER_H */
