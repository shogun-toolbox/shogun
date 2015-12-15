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

#ifndef SMIDASMINIMIZER_H
#define SMIDASMINIMIZER_H
#include <shogun/optimization/SMDMinimizer.h>
#include <shogun/lib/config.h>
namespace shogun
{

/** @brief The class implements the stochastic mirror descend (SMD) minimizer
 *
 * Shai Shalev-Shwartz and Ambuj Tewari,
 * Stochastic methods for l1 regularized loss minimization.
 * Proceedings of the 26th International Conference on Machine Learning,
 * pages 929-936, 2009. 
 */

class SMIDASMinimizer: public SMDMinimizer
{
public:
	/** Default constructor */
	SMIDASMinimizer();

	/** Constructor
	 * @param fun stochastic cost function
	 */
	SMIDASMinimizer(FirstOrderStochasticCostFunction *fun);

	/** Destructor */
	virtual ~SMIDASMinimizer();

	/** Do minimization and get the optimal value 
	 * 
	 * @return optimal value
	 */
	virtual float64_t minimize();


	/** Load the given context object to restores mutable variables
	 * Usually it is used in deserialization.
	 *
	 * @param context a context object
	 */
	virtual void load_from_context(CMinimizerContext* context)
	{
		SMDMinimizer::load_from_context(context);
		std::string key="SMIDASMinimizer::m_dual_variable";
		SGVector<float64_t> value=context->get_data_sgvector_float64(key);
		m_dual_variable=SGVector<float64_t>(value.vlen);
		std::copy(value.vector, value.vector+value.vlen,
			m_dual_variable.vector);
	}

protected:

	/** Update a context object to store mutable variables
	 *
	 * @param context a context object
	 */
	virtual void update_context(CMinimizerContext* context)
	{
		SMDMinimizer::update_context(context);
		SGVector<float64_t> value(m_dual_variable.vlen);
		std::copy(m_dual_variable.vector,
			m_dual_variable.vector+m_dual_variable.vlen,
			value.vector);
		std::string key="SMIDASMinimizer::m_dual_variable";
		context->save_data(key, value);
	}

	virtual void init_minimization();

	SGVector<float64_t> m_dual_variable;

private:
	  /* Init */
	void init();
};

}
#endif /* SMIDASMINIMIZER_H */
