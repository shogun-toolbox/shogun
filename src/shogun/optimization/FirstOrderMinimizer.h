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

#ifndef FIRSTORDERMINIMIZER_H
#define FIRSTORDERMINIMIZER_H
#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/optimization/FirstOrderCostFunction.h>
namespace shogun
{

/** @brief The first order minimizer base class.
 *
 * This class gives the interface of a minimizer
 *
 */
class CFirstOrderMinimizer : public CSGObject
{
public: 
	/** default constructor */
	CFirstOrderMinimizer()
		:CSGObject()
	{
		init();
	}
	/** constructor
	 * @param fun cost function
	 */
	CFirstOrderMinimizer(CFirstOrderCostFunction *fun)
		:CSGObject()
	{
		init();
		set_cost_function(fun);
	}

	/** destructor */
	virtual ~CFirstOrderMinimizer()
	{
		SG_UNREF(m_fun);
	}

	/** do minimization and get the optimal value 
	 * 
	 * @return optimal value
	 */
	virtual float64_t minimization()=0;

	/** does minimizer support batch update
	 * 
	 * @return whether minimizer supports batch update
	 */
	virtual bool support_batch_update() const=0;

	/** set cost function used in the minimizer
	 *
	 * @param fun the cost function
	 */
	virtual void set_cost_function(CFirstOrderCostFunction *fun)
	{
		if(m_fun!=fun)
		{
			SG_UNREF(m_fun);
			m_fun=fun;
			SG_REF(m_fun);
		}
	}
protected:
	/* cost function */
	CFirstOrderCostFunction *m_fun;

private:
	/*  init */
	void init()
	{
		m_fun=NULL;
		SG_ADD((CSGObject**)&m_fun, "cost_function","cost_function", MS_NOT_AVAILABLE);
	}
};

}
#endif
