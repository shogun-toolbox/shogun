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

#ifndef LBFGSMINIMIZER_H
#define LBFGSMINIMIZER_H
#include <shogun/optimization/FirstOrderMinimizer.h>
#include <shogun/optimization/lbfgs/lbfgs.h>
namespace shogun
{

/** linesearch method */
enum ELBFGSLineSearch
{
    MORETHUENTE=LBFGS_LINESEARCH_MORETHUENTE,
    BACKTRACKING_ARMIJO=LBFGS_LINESEARCH_BACKTRACKING_ARMIJO,
    BACKTRACKING_WOLFE=LBFGS_LINESEARCH_BACKTRACKING_WOLFE,
    BACKTRACKING_STRONG_WOLFE=LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE
};

/** @brief The class wraps the Shogun's C-style LBFGS minimizer  */
class LBFGSMinimizer: public FirstOrderMinimizer
{
public:
	/*  Default constructor */
	LBFGSMinimizer();

	/** Constructor
	 * @param fun cost function (user have to manully delete the pointer)
	 */
	LBFGSMinimizer(FirstOrderCostFunction *fun);

	/*  Destructor */
	virtual ~LBFGSMinimizer();

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

	/** set L-BFGS parameters
	 * For details please see shogun/optimization/lbfgs/lbfgs.h
	 * @param m The number of corrections to approximate the inverse hessian matrix.
	 * Default value is 100.
	 * @param max_linesearch The maximum number of trials to do line search for each L-BFGS update.
	 * Default value is 1000.
	 * @param linesearch The line search algorithm.
	 * Default value is using the backtracking with the strong Wolfe condition line search
	 * @param max_iterations The maximum number of iterations for L-BFGS update.
	 * Default value is 1000.
	 * @param delta Delta for convergence test based on the change of function value.
	 * Default value is 0.
	 * @param past Distance for delta-based convergence test.
	 * Default value is 0.
	 * @param epsilon Epsilon for convergence test based on the change of gradient.
	 * Default value is 1e-5
	 * @param min_step The minimum step of the line search.
	 * The default value is 1e-20
	 * @param max_step The maximum step of the line search.
	 * The default value is 1e+20
	 * @param ftol A parameter used in Armijo condition.
	 * Default value is 1e-4
	 * @param wolfe A parameter used in curvature condition.
	 * Default value is 0.9
	 * @param gtol A parameter used in Morethuente linesearch to control the accuracy.
	 * Default value is 0.9
	 * @param xtol The machine precision for floating-point values.
	 * Default value is 1e-16.
	 * @param orthantwise_c Coeefficient for the L1 norm of variables.
	 * This parameter should be set to zero for standard minimization problems.
	 * Setting this parameter to a positive value activates
	 * Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) method. Default value is 0.
	 * @param orthantwise_start Start index for computing L1 norm of the variables.
	 * This parameter is valid only for OWL-QN method. Default value is 0.
	 * @param orthantwise_end End index for computing L1 norm of the variables.
	 * Default value is 1.
	 */
	virtual void set_lbfgs_parameters(int m = 100,
		int max_linesearch = 1000,
		ELBFGSLineSearch linesearch = BACKTRACKING_STRONG_WOLFE,
		int max_iterations = 1000,
		float64_t delta = 0.0,
		int past = 0,
		float64_t epsilon = 1e-5,
		float64_t min_step = 1e-20,
		float64_t max_step = 1e+20,
		float64_t ftol = 1e-4,
		float64_t wolfe = 0.9,
		float64_t gtol = 0.9,
		float64_t xtol = 1e-16,
		float64_t orthantwise_c = 0.0,
		int orthantwise_start = 0,
		int orthantwise_end = 1);


	/** Return a context object which stores mutable variables
	 * Usually it is used in serialization.
	 *
	 * @return a context object
	 */
	virtual CMinimizerContext* save_to_context() {return new CMinimizerContext();}

	/** Load the given context object to restores mutable variables
	 * Usually it is used in deserialization.
	 *
	 * @param context a context object
	 */
	virtual void load_from_context(CMinimizerContext* context) {}

private:
	/** A helper function is used in the C-style LBFGS API
	 * Note that this function should be static and
	 * private. */
	static float64_t evaluate(void *obj,
		const float64_t *variable,
		float64_t *gradient,
		const int dim,
		const float64_t step);

	/** Init */
	void init();

protected:
	/** Init before minimization */
	virtual void init_minimization();

	/** The number of corrections to approximate the inverse hessian matrix.*/
	int m_m;

	/** The maximum number of trials to do line search for each L-BFGS update.*/
	int m_max_linesearch;

	/** The line search algorithm.*/
	int m_linesearch;

	/** The maximum number of iterations for L-BFGS update.*/
	int m_max_iterations;

	/** Delta for convergence test based on the change of function value.*/
	float64_t m_delta;

	/** Distance for delta-based convergence test.*/
	int m_past;

	/** Epsilon for convergence test based on the change of gradient.*/
	float64_t m_epsilon;

	/** The minimum step of the line search.*/
	float64_t m_min_step;

	/** The maximum step of the line search.*/
	float64_t m_max_step;

	/** A parameter used in Armijo condition.*/
	float64_t m_ftol;

	/** A parameter used in curvature condition.*/
	float64_t m_wolfe;

	/** A parameter used in Morethuente linesearch to control the accuracy.*/
	float64_t m_gtol;

	/** The machine precision for floating-point values.*/
	float64_t m_xtol;

	/** Coeefficient for the L1 norm of variables.*/
	float64_t m_orthantwise_c;

	/** Start index for computing L1 norm of the variables.*/
	int m_orthantwise_start;

	/** End index for computing L1 norm of the variables.*/
	int m_orthantwise_end;

	/** Target variable */
	SGVector<float64_t> m_target_variable;
};

}
#endif
