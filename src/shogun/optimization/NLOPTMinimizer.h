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


#ifndef CNLOPTMINIMIZER_H
#define CNLOPTMINIMIZER_H
#include <shogun/optimization/FirstOrderMinimizer.h>

#ifdef USE_GPL_SHOGUN
#ifdef HAVE_NLOPT
#include <shogun/optimization/nloptcommon.h>
#endif 
namespace shogun
{
/** @brief The class wraps the external NLOPT library
 *
 * This minimizer supports bound constrainted minimization
 * and unconstrainted minimization using the NLOPT library
 *
 */
class CNLOPTMinimizer: public FirstOrderMinimizer
{
public:
	/** Default constructor */
	CNLOPTMinimizer();

	/** Constructor
	 * @param fun cost function
	 */
	CNLOPTMinimizer(FirstOrderCostFunction *fun);

	/** returns the name of the class
	 *
	 * @return name CNLOPTMinimizer
	 */
	virtual const char* get_name() const { return "NLOPTMinimizer"; }


	/** Destructor */
	virtual ~CNLOPTMinimizer();

	/** Do minimization and get the optimal value 
	 * 
	 * @return optimal value
	 */
	virtual float64_t minimize();

	/** Does minimizer support batch update?
	 * 
	 * @return whether minimizer supports batch update
	 */
	virtual bool supports_batch_update() const {return true;}

#ifdef HAVE_NLOPT
	/* Set parameters used in NLOPT
	 * For details please see http://ab-initio.mit.edu/wiki/index.php/NLopt_C-plus-plus_Reference
	 *
	 * @param algorithm provided by NLOPT for minimization (e.g. LD_LBFGS denotes NLOPT_LD_LBFGS)
	 * @param max_iterations the number of cost function evaluations 
	 * @param variable_tolerance absolute tolerance on optimization parameters 
	 * @param function_tolerance absolute tolerance on function value.
	 */
	virtual void set_nlopt_parameters(ENLOPTALGORITHM algorithm=LD_LBFGS,
		float64_t max_iterations=1000,
		float64_t variable_tolerance=1e-6,
		float64_t function_tolerance=1e-6);
private:
	/* A helper function will be called by the NLOPT library
	 * Note that this function should be static and
	 * private.
	 * */
	static double nlopt_function(unsigned dim, const double* variable,
		double* gradient, void* func_data);

	static int16_t get_nlopt_algorithm_id(ENLOPTALGORITHM method);

	static nlopt_algorithm get_nlopt_algorithm(int16_t method_id)
	{
		REQUIRE(method_id>=0 && method_id<(int16_t)NLOPT_NUM_ALGORITHMS,
			"Unsupported method id (%d)\n", method_id);
		return (nlopt_algorithm) method_id;
	}

protected:

	/* Target variable */
	SGVector<float64_t> m_target_variable;

	/* Init before minimization */
	virtual void init_minimization();

	/** max number of iterations */
	float64_t m_max_iterations;

	/** absolute tolerance on optimization parameters */
	float64_t m_variable_tolerance;

	/** absolute tolerance on function value */
	float64_t m_function_tolerance;

	/** algorithm provided by NLOPT for minimization  */
	int16_t m_nlopt_algorithm_id;
#endif /* HAVE_NLOPT */

private:
	/* init */
	void init();
};

}
#endif //USE_GPL_SHOGUN
#endif /* CNLOPTMINIMIZER_H */

