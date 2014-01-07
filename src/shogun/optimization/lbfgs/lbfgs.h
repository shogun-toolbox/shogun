/*
 *      C library of Limited memory BFGS (L-BFGS).
 *
 * Copyright (c) 1990, Jorge Nocedal
 * Copyright (c) 2007-2010 Naoaki Okazaki
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef __LBFGS_H__
#define __LBFGS_H__

#include <lib/common.h>

namespace shogun
{

/**
 * \addtogroup liblbfgs_api libLBFGS API
 * @{
 *
 *  The libLBFGS API.
 */

/**
 * Return values of lbfgs().
 *
 *  Roughly speaking, a negative value indicates an error.
 */
enum {
    /** L-BFGS reaches convergence. */
    LBFGS_SUCCESS = 0,
    LBFGS_CONVERGENCE = 0,
    LBFGS_STOP,
    /** The initial variables already minimize the objective function. */
    LBFGS_ALREADY_MINIMIZED,

    /** Unknown error. */
    LBFGSERR_UNKNOWNERROR = -1024,
    /** Logic error. */
    LBFGSERR_LOGICERROR,
    /** Insufficient memory. */
    LBFGSERR_OUTOFMEMORY,
    /** The minimization process has been canceled. */
    LBFGSERR_CANCELED,
    /** Invalid number of variables specified. */
    LBFGSERR_INVALID_N,
    /** Invalid number of variables (for SSE) specified. */
    LBFGSERR_INVALID_N_SSE,
    /** The array x must be aligned to 16 (for SSE). */
    LBFGSERR_INVALID_X_SSE,
    /** Invalid parameter lbfgs_parameter_t::epsilon specified. */
    LBFGSERR_INVALID_EPSILON,
    /** Invalid parameter lbfgs_parameter_t::past specified. */
    LBFGSERR_INVALID_TESTPERIOD,
    /** Invalid parameter lbfgs_parameter_t::delta specified. */
    LBFGSERR_INVALID_DELTA,
    /** Invalid parameter lbfgs_parameter_t::linesearch specified. */
    LBFGSERR_INVALID_LINESEARCH,
    /** Invalid parameter lbfgs_parameter_t::max_step specified. */
    LBFGSERR_INVALID_MINSTEP,
    /** Invalid parameter lbfgs_parameter_t::max_step specified. */
    LBFGSERR_INVALID_MAXSTEP,
    /** Invalid parameter lbfgs_parameter_t::ftol specified. */
    LBFGSERR_INVALID_FTOL,
    /** Invalid parameter lbfgs_parameter_t::wolfe specified. */
    LBFGSERR_INVALID_WOLFE,
    /** Invalid parameter lbfgs_parameter_t::gtol specified. */
    LBFGSERR_INVALID_GTOL,
    /** Invalid parameter lbfgs_parameter_t::xtol specified. */
    LBFGSERR_INVALID_XTOL,
    /** Invalid parameter lbfgs_parameter_t::max_linesearch specified. */
    LBFGSERR_INVALID_MAXLINESEARCH,
    /** Invalid parameter lbfgs_parameter_t::orthantwise_c specified. */
    LBFGSERR_INVALID_ORTHANTWISE,
    /** Invalid parameter lbfgs_parameter_t::orthantwise_start specified. */
    LBFGSERR_INVALID_ORTHANTWISE_START,
    /** Invalid parameter lbfgs_parameter_t::orthantwise_end specified. */
    LBFGSERR_INVALID_ORTHANTWISE_END,
    /** The line-search step went out of the interval of uncertainty. */
    LBFGSERR_OUTOFINTERVAL,
    /** A logic error occurred; alternatively, the interval of uncertainty
        became too small. */
    LBFGSERR_INCORRECT_TMINMAX,
    /** A rounding error occurred; alternatively, no line-search step
        satisfies the sufficient decrease and curvature conditions. */
    LBFGSERR_ROUNDING_ERROR,
    /** The line-search step became smaller than lbfgs_parameter_t::min_step. */
    LBFGSERR_MINIMUMSTEP,
    /** The line-search step became larger than lbfgs_parameter_t::max_step. */
    LBFGSERR_MAXIMUMSTEP,
    /** The line-search routine reaches the maximum number of evaluations. */
    LBFGSERR_MAXIMUMLINESEARCH,
    /** The algorithm routine reaches the maximum number of iterations. */
    LBFGSERR_MAXIMUMITERATION,
    /** Relative width of the interval of uncertainty is at most
        lbfgs_parameter_t::xtol. */
    LBFGSERR_WIDTHTOOSMALL,
    /** A logic error (negative line-search step) occurred. */
    LBFGSERR_INVALIDPARAMETERS,
    /** The current search direction increases the objective function value. */
    LBFGSERR_INCREASEGRADIENT,
};

/**
 * Line search algorithms.
 */
enum {
    /** The default algorithm (MoreThuente method). */
    LBFGS_LINESEARCH_DEFAULT = 0,
    /** MoreThuente method proposd by More and Thuente. */
    LBFGS_LINESEARCH_MORETHUENTE = 0,
    /**
     * Backtracking method with the Armijo condition.
     *  The backtracking method finds the step length such that it satisfies
     *  the sufficient decrease (Armijo) condition,
     *    - f(x + a * d) <= f(x) + lbfgs_parameter_t::ftol * a * g(x)^T d,
     *
     *  where x is the current point, d is the current search direction, and
     *  a is the step length.
     */
    LBFGS_LINESEARCH_BACKTRACKING_ARMIJO = 1,
    /** The backtracking method with the defualt (regular Wolfe) condition. */
    LBFGS_LINESEARCH_BACKTRACKING = 2,
    /**
     * Backtracking method with regular Wolfe condition.
     *  The backtracking method finds the step length such that it satisfies
     *  both the Armijo condition (LBFGS_LINESEARCH_BACKTRACKING_ARMIJO)
     *  and the curvature condition,
     *    - g(x + a * d)^T d >= lbfgs_parameter_t::wolfe * g(x)^T d,
     *
     *  where x is the current point, d is the current search direction, and
     *  a is the step length.
     */
    LBFGS_LINESEARCH_BACKTRACKING_WOLFE = 2,
    /**
     * Backtracking method with strong Wolfe condition.
     *  The backtracking method finds the step length such that it satisfies
     *  both the Armijo condition (LBFGS_LINESEARCH_BACKTRACKING_ARMIJO)
     *  and the following condition,
     *    - |g(x + a * d)^T d| <= lbfgs_parameter_t::wolfe * |g(x)^T d|,
     *
     *  where x is the current point, d is the current search direction, and
     *  a is the step length.
     */
    LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 3,
};

/**
 * L-BFGS optimization parameters.
 *  Call lbfgs_parameter_init() function to initialize parameters to the
 *  default values.
 */
typedef struct {
    /**
     * The number of corrections to approximate the inverse hessian matrix.
     *  The L-BFGS routine stores the computation results of previous \ref m
     *  iterations to approximate the inverse hessian matrix of the current
     *  iteration. This parameter controls the size of the limited memories
     *  (corrections). The default value is \c 6. Values less than \c 3 are
     *  not recommended. Large values will result in excessive computing time.
     */
    int             m;

    /**
     * Epsilon for convergence test.
     *  This parameter determines the accuracy with which the solution is to
     *  be found. A minimization terminates when
     *      ||g|| < \ref epsilon * max(1, ||x||),
     *  where ||.|| denotes the Euclidean (L2) norm. The default value is
     *  \c 1e-5.
     */
    float64_t epsilon;

    /**
     * Distance for delta-based convergence test.
     *  This parameter determines the distance, in iterations, to compute
     *  the rate of decrease of the objective function. If the value of this
     *  parameter is zero, the library does not perform the delta-based
     *  convergence test. The default value is \c 0.
     */
    int             past;

    /**
     * Delta for convergence test.
     *  This parameter determines the minimum rate of decrease of the
     *  objective function. The library stops iterations when the
     *  following condition is met:
     *      (f' - f) / f < \ref delta,
     *  where f' is the objective value of \ref past iterations ago, and f is
     *  the objective value of the current iteration.
     *  The default value is \c 0.
     */
    float64_t delta;

    /**
     * The maximum number of iterations.
     *  The lbfgs() function terminates an optimization process with
     *  LBFGSERR_MAXIMUMITERATION status code when the iteration count
     *  exceedes this parameter. Setting this parameter to zero continues an
     *  optimization process until a convergence or error. The default value
     *  is \c 0.
     */
    int             max_iterations;

    /**
     * The line search algorithm.
     *  This parameter specifies a line search algorithm to be used by the
     *  L-BFGS routine.
     */
    int             linesearch;

    /**
     * The maximum number of trials for the line search.
     *  This parameter controls the number of function and gradients evaluations
     *  per iteration for the line search routine. The default value is \c 20.
     */
    int             max_linesearch;

    /**
     * The minimum step of the line search routine.
     *  The default value is \c 1e-20. This value need not be modified unless
     *  the exponents are too large for the machine being used, or unless the
     *  problem is extremely badly scaled (in which case the exponents should
     *  be increased).
     */
    float64_t min_step;

    /**
     * The maximum step of the line search.
     *  The default value is \c 1e+20. This value need not be modified unless
     *  the exponents are too large for the machine being used, or unless the
     *  problem is extremely badly scaled (in which case the exponents should
     *  be increased).
     */
    float64_t max_step;

    /**
     * A parameter to control the accuracy of the line search routine.
     *  The default value is \c 1e-4. This parameter should be greater
     *  than zero and smaller than \c 0.5.
     */
    float64_t ftol;

    /**
     * A coefficient for the Wolfe condition.
     *  This parameter is valid only when the backtracking line-search
     *  algorithm is used with the Wolfe condition,
     *  LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE or
     *  LBFGS_LINESEARCH_BACKTRACKING_WOLFE .
     *  The default value is \c 0.9. This parameter should be greater
     *  the \ref ftol parameter and smaller than \c 1.0.
     */
    float64_t wolfe;

    /**
     * A parameter to control the accuracy of the line search routine.
     *  The default value is \c 0.9. If the function and gradient
     *  evaluations are inexpensive with respect to the cost of the
     *  iteration (which is sometimes the case when solving very large
     *  problems) it may be advantageous to set this parameter to a small
     *  value. A typical small value is \c 0.1. This parameter shuold be
     *  greater than the \ref ftol parameter (\c 1e-4) and smaller than
     *  \c 1.0.
     */
    float64_t gtol;

    /**
     * The machine precision for floating-point values.
     *  This parameter must be a positive value set by a client program to
     *  estimate the machine precision. The line search routine will terminate
     *  with the status code (LBFGSERR_ROUNDING_ERROR) if the relative width
     *  of the interval of uncertainty is less than this parameter.
     */
    float64_t xtol;

    /**
     * Coeefficient for the L1 norm of variables.
     *  This parameter should be set to zero for standard minimization
     *  problems. Setting this parameter to a positive value activates
     *  Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) method, which
     *  minimizes the objective function F(x) combined with the L1 norm |x|
     *  of the variables, {F(x) + C |x|}. This parameter is the coeefficient
     *  for the |x|, i.e., C. As the L1 norm |x| is not differentiable at
     *  zero, the library modifies function and gradient evaluations from
     *  a client program suitably; a client program thus have only to return
     *  the function value F(x) and gradients G(x) as usual. The default value
     *  is zero.
     */
    float64_t orthantwise_c;

    /**
     * Start index for computing L1 norm of the variables.
     *  This parameter is valid only for OWL-QN method
     *  (i.e., \ref orthantwise_c != 0). This parameter b (0 <= b < N)
     *  specifies the index number from which the library computes the
     *  L1 norm of the variables x,
     *      |x| := |x_{b}| + |x_{b+1}| + ... + |x_{N}| .
     *  In other words, variables x_1, ..., x_{b-1} are not used for
     *  computing the L1 norm. Setting b (0 < b < N), one can protect
     *  variables, x_1, ..., x_{b-1} (e.g., a bias term of logistic
     *  regression) from being regularized. The default value is zero.
     */
    int             orthantwise_start;

    /**
     * End index for computing L1 norm of the variables.
     *  This parameter is valid only for OWL-QN method
     *  (i.e., \ref orthantwise_c != 0). This parameter e (0 < e <= N)
     *  specifies the index number at which the library stops computing the
     *  L1 norm of the variables x,
     */
    int             orthantwise_end;
} lbfgs_parameter_t;


/**
 * Callback interface to provide objective function and gradient evaluations.
 *
 *  The lbfgs() function call this function to obtain the values of objective
 *  function and its gradients when needed. A client program must implement
 *  this function to evaluate the values of the objective function and its
 *  gradients, given current values of variables.
 *
 *  @param  instance    The user data sent for lbfgs() function by the client.
 *  @param  x           The current values of variables.
 *  @param  g           The gradient vector. The callback function must compute
 *                      the gradient values for the current variables.
 *  @param  n           The number of variables.
 *  @param  step        The current step of the line search routine.
 *  @retval float64_t The value of the objective function for the current
 *                          variables.
 */
typedef float64_t (*lbfgs_evaluate_t)(
    void *instance,
    const float64_t *x,
    float64_t *g,
    const int n,
    const float64_t step
    );

/**
 * Callback interface to receive the progress of the optimization process.
 *
 *  The lbfgs() function call this function for each iteration. Implementing
 *  this function, a client program can store or display the current progress
 *  of the optimization process.
 *
 *  @param  instance    The user data sent for lbfgs() function by the client.
 *  @param  x           The current values of variables.
 *  @param  g           The current gradient values of variables.
 *  @param  fx          The current value of the objective function.
 *  @param  xnorm       The Euclidean norm of the variables.
 *  @param  gnorm       The Euclidean norm of the gradients.
 *  @param  step        The line-search step used for this iteration.
 *  @param  n           The number of variables.
 *  @param  k           The iteration count.
 *  @param  ls          The number of evaluations called for this iteration.
 *  @retval int         Zero to continue the optimization process. Returning a
 *                      non-zero value will cancel the optimization process.
 */
typedef int (*lbfgs_progress_t)(
    void *instance,
    const float64_t *x,
    const float64_t *g,
    const float64_t fx,
    const float64_t xnorm,
    const float64_t gnorm,
    const float64_t step,
    int n,
    int k,
    int ls
    );

/*
A user must implement a function compatible with ::lbfgs_evaluate_t (evaluation
callback) and pass the pointer to the callback function to lbfgs() arguments.
Similarly, a user can implement a function compatible with ::lbfgs_progress_t
(progress callback) to obtain the current progress (e.g., variables, function
value, ||G||, etc) and to cancel the iteration process if necessary.
Implementation of a progress callback is optional: a user can pass \c NULL if
progress notification is not necessary.

In addition, a user must preserve two requirements:
    - The number of variables must be multiples of 16 (this is not 4).
    - The memory block of variable array ::x must be aligned to 16.

This algorithm terminates an optimization
when:

    ||G|| < \epsilon \cdot \max(1, ||x||) .

In this formula, ||.|| denotes the Euclidean norm.
*/

/**
 * Start a L-BFGS optimization.
 *
 *  @param  n           The number of variables.
 *  @param  x           The array of variables. A client program can set
 *                      default values for the optimization and receive the
 *                      optimization result through this array. This array
 *                      must be allocated by lbfgs_malloc function
 *                      for libLBFGS built with SSE/SSE2 optimization routine
 *                      enabled. The library built without SSE/SSE2
 *                      optimization does not have such a requirement.
 *  @param  ptr_fx      The pointer to the variable that receives the final
 *                      value of the objective function for the variables.
 *                      This argument can be set to \c NULL if the final
 *                      value of the objective function is unnecessary.
 *  @param  proc_evaluate   The callback function to provide function and
 *                          gradient evaluations given a current values of
 *                          variables. A client program must implement a
 *                          callback function compatible with \ref
 *                          lbfgs_evaluate_t and pass the pointer to the
 *                          callback function.
 *  @param  proc_progress   The callback function to receive the progress
 *                          (the number of iterations, the current value of
 *                          the objective function) of the minimization
 *                          process. This argument can be set to \c NULL if
 *                          a progress report is unnecessary.
 *  @param  instance    A user data for the client program. The callback
 *                      functions will receive the value of this argument.
 *  @param  param       The pointer to a structure representing parameters for
 *                      L-BFGS optimization. A client program can set this
 *                      parameter to \c NULL to use the default parameters.
 *                      Call lbfgs_parameter_init() function to fill a
 *                      structure with the default values.
 *  @retval int         The status code. This function returns zero if the
 *                      minimization process terminates without an error. A
 *                      non-zero value indicates an error.
 */
int lbfgs(
    int n,
    float64_t *x,
    float64_t *ptr_fx,
    lbfgs_evaluate_t proc_evaluate,
    lbfgs_progress_t proc_progress,
    void *instance,
    lbfgs_parameter_t *param
    );

/**
 * Initialize L-BFGS parameters to the default values.
 *
 *  Call this function to fill a parameter structure with the default values
 *  and overwrite parameter values if necessary.
 *
 *  @param  param       The pointer to the parameter structure.
 */
void lbfgs_parameter_init(lbfgs_parameter_t *param);

/** @} */

} // namespace shogun

#endif/*__LBFGS_H__*/
