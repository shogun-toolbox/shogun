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

#ifndef PNORMMAPPINGFUNCTION_H
#define PNORMMAPPINGFUNCTION_H
#include <shogun/optimization/MappingFunction.h>
#include <shogun/mathematics/Math.h>
namespace shogun
{
/** @brief This implements the P-norm mapping function
 *
 * Gentile, Claudio. "The robustness of the p-norm algorithms."
 * Machine Learning 53.3 (2003): 265-299.
 */
class PNormMappingFunction: public MappingFunction
{
public:
	PNormMappingFunction()
		:MappingFunction()
	{
		init();
	}
	virtual ~PNormMappingFunction() {}

	virtual void set_norm(float64_t p)
	{
		if(p<2.0)
		{
			SG_SWARNING("The norm (%f) should not be less than 2.0 and we use p=2.0 in this case\n", p);
		}
		else
			m_p=p;
	}

	virtual SGVector<float64_t> get_dual_variable(SGVector<float64_t> variable)
	{
		SGVector<float64_t> dual_variable(variable.vlen);
		float64_t q=1.0/(1.0-1.0/m_p);
		transformation(variable, dual_variable, q);
		return dual_variable;
	}

	virtual void update_variable(SGVector<float64_t> variable, SGVector<float64_t> dual_variable)
	{
		transformation(dual_variable, variable, m_p);
	}

	/** Update a context object to store mutable variables
	 * used in learning rate
	 *
	 * This method will be called by
	 * SMDMinimizer::update_context()
	 *
	 * @param context, a context object
	 */
	virtual void update_context(CMinimizerContext* context)
	{
		REQUIRE(context, "Contest must not NULL\n");
	}

	/** Load the given context object to restore mutable variables
	 *
	 * This method will be called by
	 * SMDMinimizer::load_from_context(CMinimizerContext* context)
	 * @param context, a context object
	 */
	virtual void load_from_context(CMinimizerContext* context)
	{
		REQUIRE(context, "Contest must not NULL\n");
	}
protected:
	float64_t m_p;

	virtual void transformation(SGVector<float64_t> input, SGVector<float64_t> output, float64_t degree)
	{
		REQUIRE(input.vlen==output.vlen,"");
		float64_t scale=0.0;
		for(index_t idx=0; idx<input.vlen; idx++)
		{
			scale += CMath::pow(CMath::abs(input[idx]),degree);
			if (input[idx] >= 0.0)
				output[idx]=CMath::pow(input[idx],degree-1);
			else
				output[idx]=-CMath::pow(-input[idx],degree-1);
		}
		scale=CMath::pow(scale,1.0-2.0/degree);
		for(index_t idx=0; idx<input.vlen; idx++)
			output[idx]/=scale;
	}

private:
	void init()
	{
		m_p=2.0;
	}
};

}

#endif
