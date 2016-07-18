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
namespace shogun
{
/** @brief This implements the P-norm mapping/projection function
 *
 *
 * Reference:
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


	/** returns the name of the class
	 *
	 * @return name PNormMappingFunction
	 */
	virtual const char* get_name() const { return "PNormMappingFunction"; }


	/** Get the degree of the Norm   
	 * @param p degree of the norm
	 */
	virtual void set_norm(float64_t p);

	/** Get dual variable
	 *
	 * @param variable primal variable 
	 * @return dual variable 
	 *
	 */
	virtual SGVector<float64_t> get_dual_variable(SGVector<float64_t> variable);
	
	/** Update primal variable in place given dual variable
	 *
	 * @param variable primal variable to be updated
	 * @param dual_variable dual variable are known
	 *
	 */
	virtual void update_variable(SGVector<float64_t> variable, SGVector<float64_t> dual_variable);
protected:
	/** P-norm  */
	float64_t m_p;

	/** Project the input variable 
	 *
	 * @param input input variable
	 * @param output store the result
	 * @param degree the parameter of the projection
	 */
	virtual void projection(SGVector<float64_t> input, SGVector<float64_t> output, float64_t degree);

private:
	void init();
};

}

#endif
