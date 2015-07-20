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

#ifndef NLOPTMINIMIZER_H
#define NLOPTMINIMIZER_H
#include <shogun/optimization/WrappedMinimizer.h>

#ifdef HAVE_NLOPT
#include <nlopt.h>
#endif 
namespace shogun
{

/* The class wraps the nlopt minimizer
 *
 * This minimizer support bound constrainted minimization and unconstrainted minimization.
 *
 */
class CNLOPTMinimizer: public CWrappedMinimizer
{
public:
	/** default constructor */
	CNLOPTMinimizer();

	/** constructor
	 * @param fun cost function
	 */
	CNLOPTMinimizer(CFirstOrderCostFunction *fun);

	/** destructor */
	virtual ~CNLOPTMinimizer();

	/** do minimization and get the optimal value 
	 * 
	 * @return optimal value
	 */
	virtual float64_t minimization();

	/** does minimizer support batch update
	 * 
	 * @return whether minimizer supports batch update
	 */
	virtual bool support_batch_update() const {return true;}

	/** return the name of a minimizer.
	 *
	 *  @return NLOPTMinimizer 
	 */
	virtual const char* get_name() const {return "NLOPTMinimizer";}

#ifdef HAVE_NLOPT
	/* set L-BFGS parameters
	 * For details please see http://ab-initio.mit.edu/wiki/index.php/NLopt_C-plus-plus_Reference
	 *
	 * @param algorithm provided by NLOPT for minimization
	 * @param max_iterations the number of cost function evaluations 
	 * @param variable_tolerance absolute tolerance on optimization parameters 
	 * @param function_tolerance absolute tolerance on function value.
	 */
	virtual void set_nlopt_parameters(nlopt_algorithm algorithm=NLOPT_LD_LBFGS,
		float64_t max_iterations=1000,
		float64_t variable_tolerance=1e-6,
		float64_t function_tolerance=1e-6);
private:
	static double nlopt_function(unsigned dim, const double* variable,
		double* gradient, void* func_data);

	/** max number of iterations */
	float64_t m_max_iterations;

	/** absolute tolerance on optimization parameters */
	float64_t m_variable_tolerance;

	/** absolute tolerance on function value */
	float64_t m_function_tolerance;

	/** algorithm provided by NLOPT for minimization  */
	nlopt_algorithm m_nlopt_algorithm;
#endif /* HAVE_NLOPT */

private:
	/* init */
	void init();
};

}
#endif /* NLOPTMINIMIZER_H */
