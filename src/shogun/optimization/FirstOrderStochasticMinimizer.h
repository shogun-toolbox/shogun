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
#ifndef FIRSTORDERSTOCHASTICMINIMIZER_H
#define FIRSTORDERSTOCHASTICMINIMIZER_H
#include <shogun/optimization/FirstOrderMinimizer.h>
#include <shogun/optimization/FirstOrderStochasticCostFunction.h>
#include <shogun/optimization/DescendUpdater.h>
namespace shogun
{

/** @brief The base class for stochastic first-order gradient-based minimizers.
 *
 *
 */
class CFirstOrderStochasticMinimizer: public CFirstOrderMinimizer
{
public:
	/** default constructor */
	CFirstOrderStochasticMinimizer()
		:CFirstOrderMinimizer()
	{
		init();
	}

	/** constructor
	 * @param fun cost function
	 */
	CFirstOrderStochasticMinimizer(CFirstOrderStochasticCostFunction *fun)
		:CFirstOrderMinimizer(fun)
	{
		init();
	}

	/** destructor
	 */
	virtual ~CFirstOrderStochasticMinimizer(){}

	/** does minimizer support batch update
	 * 
	 * @return whether minimizer supports batch update
	 */
	virtual bool supports_batch_update() const {return false;}

	/** set a gradient updater
	 *
	 * @param gradient_updater the gradient_updater
	 */
	virtual void set_gradient_updater(CDescendUpdater* gradient_updater)
	{
		//REQUIRE(gradient_updater, "gradient_updater must set\n");

		m_gradient_updater=gradient_updater;
	}

	/** Do minimization and get the optimal value 
	 * 
	 * @return optimal value
	 */
	virtual float64_t minimize()=0;

	/** set the number of times to go through all data points
	 * For example, num_passes=1 means go through all data points once.
	 *
	 * @param num_passes the number of times 
	 */
	virtual void set_number_passes(int32_t num_passes)
	{
		//REQUIRE(num_passes>0, "The number (%d) to go through data must be positive\n", num_passes);
		m_num_passes=num_passes;
	}

	virtual void load_from_context(CMinimizerContext* context)
	{
		//REQUIRE(context,"Context must set\n");
		//REQUIRE(m_gradient_updater,"");
		m_gradient_updater->load_from_context(context);
	}

	virtual CMinimizerContext* save_to_context()
	{
		//REQUIRE(m_gradient_updater,"");
		CMinimizerContext* result=new CMinimizerContext();
		m_gradient_updater->update_context(result);
		return result;
	}
protected:
	/*  the gradient update step */
	CDescendUpdater* m_gradient_updater;

	/* init the minimization process*/
	virtual void init_minimization()
	{
		//REQUIRE(m_fun,"cost function must set\n");
		//REQUIRE(m_gradient_updater,"gradient_updater must set\n");
		//REQUIRE(m_num_passes>0, "The number (%d) to go through data must set\n");
		m_cur_passes=0;
	}

	int32_t m_num_passes;
	int32_t m_cur_passes;
private:
	/* init */
	void init()
	{
		m_gradient_updater=NULL;
		m_num_passes=0;
		m_cur_passes=0;
	}

};

}
#endif /* FIRSTORDERSTOCHASTICMINIMIZER_H */
