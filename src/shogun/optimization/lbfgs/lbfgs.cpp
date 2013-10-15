/*
 *      Limited memory BFGS (L-BFGS).
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

/* $Id$ */

/*
This library is a C port of the FORTRAN implementation of Limited-memory
Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) method written by Jorge Nocedal.
The original FORTRAN source code is available at:
http://www.ece.northwestern.edu/~nocedal/lbfgs.html

The L-BFGS algorithm is described in:
    - Jorge Nocedal.
      Updating Quasi-Newton Matrices with Limited Storage.
      <i>Mathematics of Computation</i>, Vol. 35, No. 151, pp. 773--782, 1980.
    - Dong C. Liu and Jorge Nocedal.
      On the limited memory BFGS method for large scale optimization.
      <i>Mathematical Programming</i> B, Vol. 45, No. 3, pp. 503-528, 1989.

The line search algorithms used in this implementation are described in:
    - John E. Dennis and Robert B. Schnabel.
      <i>Numerical Methods for Unconstrained Optimization and Nonlinear
      Equations</i>, Englewood Cliffs, 1983.
    - Jorge J. More and David J. Thuente.
      Line search algorithm with guaranteed sufficient decrease.
      <i>ACM Transactions on Mathematical Software (TOMS)</i>, Vol. 20, No. 3,
      pp. 286-307, 1994.

This library also implements Orthant-Wise Limited-memory Quasi-Newton (OWL-QN)
method presented in:
    - Galen Andrew and Jianfeng Gao.
      Scalable training of L1-regularized log-linear models.
      In <i>Proceedings of the 24th International Conference on Machine
      Learning (ICML 2007)</i>, pp. 33-40, 2007.

I would like to thank the original author, Jorge Nocedal, who has been
distributing the effieicnt and explanatory implementation in an open source
licence.
*/

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <shogun/optimization/lbfgs/lbfgs.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/common.h>

namespace shogun
{

#define min2(a, b)      ((a) <= (b) ? (a) : (b))
#define max2(a, b)      ((a) >= (b) ? (a) : (b))
#define max3(a, b, c)   max2(max2((a), (b)), (c));

struct tag_callback_data {
    int32_t n;
    void *instance;
    lbfgs_evaluate_t proc_evaluate;
    lbfgs_progress_t proc_progress;
};
typedef struct tag_callback_data callback_data_t;

struct tag_iteration_data {
    float64_t alpha;
    float64_t *s;     /* [n] */
    float64_t *y;     /* [n] */
    float64_t ys;     /* vecdot(y, s) */
};
typedef struct tag_iteration_data iteration_data_t;

static const lbfgs_parameter_t _defparam = {
    6, 1e-5, 0, 1e-5,
    0, LBFGS_LINESEARCH_DEFAULT, 40,
    1e-20, 1e20, 1e-4, 0.9, 0.9, 1.0e-16,
    0.0, 0, -1,
};

/* Forward function declarations. */

typedef int32_t (*line_search_proc)(
    int32_t n,
    float64_t *x,
    float64_t *f,
    float64_t *g,
    float64_t *s,
    float64_t *stp,
    const float64_t* xp,
    const float64_t* gp,
    float64_t *wa,
    callback_data_t *cd,
    const lbfgs_parameter_t *param
    );

static int32_t line_search_backtracking(
    int32_t n,
    float64_t *x,
    float64_t *f,
    float64_t *g,
    float64_t *s,
    float64_t *stp,
    const float64_t* xp,
    const float64_t* gp,
    float64_t *wa,
    callback_data_t *cd,
    const lbfgs_parameter_t *param
    );

static int32_t line_search_backtracking_owlqn(
    int32_t n,
    float64_t *x,
    float64_t *f,
    float64_t *g,
    float64_t *s,
    float64_t *stp,
    const float64_t* xp,
    const float64_t* gp,
    float64_t *wp,
    callback_data_t *cd,
    const lbfgs_parameter_t *param
    );

static int32_t line_search_morethuente(
    int32_t n,
    float64_t *x,
    float64_t *f,
    float64_t *g,
    float64_t *s,
    float64_t *stp,
    const float64_t* xp,
    const float64_t* gp,
    float64_t *wa,
    callback_data_t *cd,
    const lbfgs_parameter_t *param
    );

static int32_t update_trial_interval(
    float64_t *x,
    float64_t *fx,
    float64_t *dx,
    float64_t *y,
    float64_t *fy,
    float64_t *dy,
    float64_t *t,
    float64_t *ft,
    float64_t *dt,
    const float64_t tmin,
    const float64_t tmax,
    int32_t *brackt
    );

static float64_t owlqn_x1norm(
    const float64_t* x,
    const int32_t start,
    const int32_t n
    );

static void owlqn_pseudo_gradient(
    float64_t* pg,
    const float64_t* x,
    const float64_t* g,
    const int32_t n,
    const float64_t c,
    const int32_t start,
    const int32_t end
    );

static void owlqn_project(
    float64_t* d,
    const float64_t* sign,
    const int32_t start,
    const int32_t end
    );


void lbfgs_parameter_init(lbfgs_parameter_t *param)
{
    memcpy(param, &_defparam, sizeof(*param));
}

int32_t lbfgs(
    int32_t n,
    float64_t *x,
    float64_t *ptr_fx,
    lbfgs_evaluate_t proc_evaluate,
    lbfgs_progress_t proc_progress,
    void *instance,
    lbfgs_parameter_t *_param
    )
{
    int32_t ret;
    int32_t i, j, k, ls, end, bound;
    float64_t step;

    /* Constant parameters and their default values. */
    lbfgs_parameter_t param = (_param != NULL) ? (*_param) : _defparam;
    const int32_t m = param.m;

    float64_t *xp = NULL;
    float64_t *g = NULL, *gp = NULL, *pg = NULL;
    float64_t *d = NULL, *w = NULL, *pf = NULL;
    iteration_data_t *lm = NULL, *it = NULL;
    float64_t ys, yy;
    float64_t xnorm, gnorm, beta;
    float64_t fx = 0.;
    float64_t rate = 0.;
    line_search_proc linesearch = line_search_morethuente;

    /* Construct a callback data. */
    callback_data_t cd;
    cd.n = n;
    cd.instance = instance;
    cd.proc_evaluate = proc_evaluate;
    cd.proc_progress = proc_progress;

    /* Check the input parameters for errors. */
    if (n <= 0) {
        return LBFGSERR_INVALID_N;
    }
    if (param.epsilon < 0.) {
        return LBFGSERR_INVALID_EPSILON;
    }
    if (param.past < 0) {
        return LBFGSERR_INVALID_TESTPERIOD;
    }
    if (param.delta < 0.) {
        return LBFGSERR_INVALID_DELTA;
    }
    if (param.min_step < 0.) {
        return LBFGSERR_INVALID_MINSTEP;
    }
    if (param.max_step < param.min_step) {
        return LBFGSERR_INVALID_MAXSTEP;
    }
    if (param.ftol < 0.) {
        return LBFGSERR_INVALID_FTOL;
    }
    if (param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE ||
        param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE) {
        if (param.wolfe <= param.ftol || 1. <= param.wolfe) {
            return LBFGSERR_INVALID_WOLFE;
        }
    }
    if (param.gtol < 0.) {
        return LBFGSERR_INVALID_GTOL;
    }
    if (param.xtol < 0.) {
        return LBFGSERR_INVALID_XTOL;
    }
    if (param.max_linesearch <= 0) {
        return LBFGSERR_INVALID_MAXLINESEARCH;
    }
    if (param.orthantwise_c < 0.) {
        return LBFGSERR_INVALID_ORTHANTWISE;
    }
    if (param.orthantwise_start < 0 || n < param.orthantwise_start) {
        return LBFGSERR_INVALID_ORTHANTWISE_START;
    }
    if (param.orthantwise_end < 0) {
        param.orthantwise_end = n;
    }
    if (n < param.orthantwise_end) {
        return LBFGSERR_INVALID_ORTHANTWISE_END;
    }
    if (param.orthantwise_c != 0.) {
        switch (param.linesearch) {
        case LBFGS_LINESEARCH_BACKTRACKING:
            linesearch = line_search_backtracking_owlqn;
            break;
        default:
            /* Only the backtracking method is available. */
            return LBFGSERR_INVALID_LINESEARCH;
        }
    } else {
        switch (param.linesearch) {
        case LBFGS_LINESEARCH_MORETHUENTE:
            linesearch = line_search_morethuente;
            break;
        case LBFGS_LINESEARCH_BACKTRACKING_ARMIJO:
        case LBFGS_LINESEARCH_BACKTRACKING_WOLFE:
        case LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE:
            linesearch = line_search_backtracking;
            break;
        default:
            return LBFGSERR_INVALID_LINESEARCH;
        }
    }

    /* Allocate working space. */
    xp = SG_CALLOC(float64_t, n);
    g = SG_CALLOC(float64_t, n);
    gp = SG_CALLOC(float64_t, n);
    d = SG_CALLOC(float64_t, n);
    w = SG_CALLOC(float64_t, n);
    if (xp == NULL || g == NULL || gp == NULL || d == NULL || w == NULL) {
        ret = LBFGSERR_OUTOFMEMORY;
        goto lbfgs_exit;
    }

    if (param.orthantwise_c != 0.) {
        /* Allocate working space for OW-LQN. */
        pg = SG_CALLOC(float64_t, n);
        if (pg == NULL) {
            ret = LBFGSERR_OUTOFMEMORY;
            goto lbfgs_exit;
        }
    }

    /* Allocate limited memory storage. */
    lm = SG_CALLOC(iteration_data_t, m);
    if (lm == NULL) {
        ret = LBFGSERR_OUTOFMEMORY;
        goto lbfgs_exit;
    }

    /* Initialize the limited memory. */
    for (i = 0;i < m;++i) {
        it = &lm[i];
        it->alpha = 0;
        it->ys = 0;
        it->s = SG_CALLOC(float64_t, n);
        it->y = SG_CALLOC(float64_t, n);
        if (it->s == NULL || it->y == NULL) {
            ret = LBFGSERR_OUTOFMEMORY;
            goto lbfgs_exit;
        }
    }

    /* Allocate an array for storing previous values of the objective function. */
    if (0 < param.past) {
        pf = SG_CALLOC(float64_t, param.past);
    }

    /* Evaluate the function value and its gradient. */
    fx = cd.proc_evaluate(cd.instance, x, g, cd.n, 0);
    if (0. != param.orthantwise_c) {
        /* Compute the L1 norm of the variable and add it to the object value. */
        xnorm = owlqn_x1norm(x, param.orthantwise_start, param.orthantwise_end);
        fx += xnorm * param.orthantwise_c;
        owlqn_pseudo_gradient(
            pg, x, g, n,
            param.orthantwise_c, param.orthantwise_start, param.orthantwise_end
            );
    }

    /* Store the initial value of the objective function. */
    if (pf != NULL) {
        pf[0] = fx;
    }

    /*
        Compute the direction;
        we assume the initial hessian matrix H_0 as the identity matrix.
     */
    if (param.orthantwise_c == 0.) {
		std::copy(g,g+n,d);
		SGVector<float64_t>::scale_vector(-1, d, n);
    } else {
		std::copy(pg,pg+n,d);
		SGVector<float64_t>::scale_vector(-1, d, n);
    }

    /*
       Make sure that the initial variables are not a minimizer.
     */
	xnorm = SGVector<float64_t>::twonorm(x, n);
    if (param.orthantwise_c == 0.) {
		gnorm = SGVector<float64_t>::twonorm(g, n);
    } else {
		gnorm = SGVector<float64_t>::twonorm(pg, n);
    }
    if (xnorm < 1.0) xnorm = 1.0;
    if (gnorm / xnorm <= param.epsilon) {
        ret = LBFGS_ALREADY_MINIMIZED;
        goto lbfgs_exit;
    }

    /* Compute the initial step:
        step = 1.0 / sqrt(vecdot(d, d, n))
     */
	step = 1.0 / SGVector<float64_t>::twonorm(d, n);

    k = 1;
    end = 0;
    for (;;) {
        /* Store the current position and gradient vectors. */
		std::copy(x,x+n,xp);
		std::copy(g,g+n,gp);

        /* Search for an optimal step. */
        if (param.orthantwise_c == 0.) {
            ls = linesearch(n, x, &fx, g, d, &step, xp, gp, w, &cd, &param);
        } else {
            ls = linesearch(n, x, &fx, g, d, &step, xp, pg, w, &cd, &param);
            owlqn_pseudo_gradient(
                pg, x, g, n,
                param.orthantwise_c, param.orthantwise_start, param.orthantwise_end
                );
        }
        if (ls < 0) {
            /* Revert to the previous point. */
			std::copy(xp,xp+n,x);
			std::copy(gp,gp+n,g);
            ret = ls;
            goto lbfgs_exit;
        }

        /* Compute x and g norms. */
		xnorm = SGVector<float64_t>::twonorm(x, n);
        if (param.orthantwise_c == 0.) {
			gnorm = SGVector<float64_t>::twonorm(g, n);
        } else {
			gnorm = SGVector<float64_t>::twonorm(pg, n);
        }

        /* Report the progress. */
        if (cd.proc_progress) {
            if ((ret = cd.proc_progress(cd.instance, x, g, fx, xnorm, gnorm, step, cd.n, k, ls))) {
                goto lbfgs_exit;
            }
        }

        /*
            Convergence test.
            The criterion is given by the following formula:
                |g(x)| / \max2(1, |x|) < \epsilon
         */
        if (xnorm < 1.0) xnorm = 1.0;
        if (gnorm / xnorm <= param.epsilon) {
            /* Convergence. */
            ret = LBFGS_SUCCESS;
            break;
        }

        /*
            Test for stopping criterion.
            The criterion is given by the following formula:
                (f(past_x) - f(x)) / f(x) < \delta
         */
        if (pf != NULL) {
            /* We don't test the stopping criterion while k < past. */
            if (param.past <= k) {
                /* Compute the relative improvement from the past. */
                rate = (pf[k % param.past] - fx) / fx;

                /* The stopping criterion. */
                if (rate < param.delta) {
                    ret = LBFGS_STOP;
                    break;
                }
            }

            /* Store the current value of the objective function. */
            pf[k % param.past] = fx;
        }

        if (param.max_iterations != 0 && param.max_iterations < k+1) {
            /* Maximum number of iterations. */
            ret = LBFGSERR_MAXIMUMITERATION;
            break;
        }

        /*
            Update vectors s and y:
                s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
                y_{k+1} = g_{k+1} - g_{k}.
         */
        it = &lm[end];
		SGVector<float64_t>::add(it->s, 1, x, -1, xp, n);
		SGVector<float64_t>::add(it->y, 1, g, -1, gp, n);

        /*
            Compute scalars ys and yy:
                ys = y^t \cdot s = 1 / \rho.
                yy = y^t \cdot y.
            Notice that yy is used for scaling the hessian matrix H_0 (Cholesky factor).
         */
		ys = SGVector<float64_t>::dot(it->y, it->s, n);
		yy = SGVector<float64_t>::dot(it->y, it->y, n);
        it->ys = ys;

        /*
            Recursive formula to compute dir = -(H \cdot g).
                This is described in page 779 of:
                Jorge Nocedal.
                Updating Quasi-Newton Matrices with Limited Storage.
                Mathematics of Computation, Vol. 35, No. 151,
                pp. 773--782, 1980.
         */
        bound = (m <= k) ? m : k;
        ++k;
        end = (end + 1) % m;

        /* Compute the steepest direction. */
        if (param.orthantwise_c == 0.) {
            /* Compute the negative of gradients. */
			std::copy(g, g+n, d);
			SGVector<float64_t>::scale_vector(-1, d, n);
        } else {
			std::copy(pg, pg+n, d);
			SGVector<float64_t>::scale_vector(-1, d, n);
        }

        j = end;
        for (i = 0;i < bound;++i) {
            j = (j + m - 1) % m;    /* if (--j == -1) j = m-1; */
            it = &lm[j];
            /* \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}. */
			it->alpha = SGVector<float64_t>::dot(it->s, d, n);
            it->alpha /= it->ys;
            /* q_{i} = q_{i+1} - \alpha_{i} y_{i}. */
			SGVector<float64_t>::add(d, 1, d, -it->alpha, it->y, n);
        }

		SGVector<float64_t>::scale_vector(ys / yy, d, n);

        for (i = 0;i < bound;++i) {
            it = &lm[j];
            /* \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}. */
			beta = SGVector<float64_t>::dot(it->y, d, n);
            beta /= it->ys;
            /* \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}. */
			SGVector<float64_t>::add(d, 1, d, it->alpha-beta, it->s, n);
            j = (j + 1) % m;        /* if (++j == m) j = 0; */
        }

        /*
            Constrain the search direction for orthant-wise updates.
         */
        if (param.orthantwise_c != 0.) {
            for (i = param.orthantwise_start;i < param.orthantwise_end;++i) {
                if (d[i] * pg[i] >= 0) {
                    d[i] = 0;
                }
            }
        }

        /*
            Now the search direction d is ready. We try step = 1 first.
         */
        step = 1.0;
    }

lbfgs_exit:
    /* Return the final value of the objective function. */
    if (ptr_fx != NULL) {
        *ptr_fx = fx;
    }

    SG_FREE(pf);

    /* Free memory blocks used by this function. */
    if (lm != NULL) {
        for (i = 0;i < m;++i) {
            SG_FREE(lm[i].s);
            SG_FREE(lm[i].y);
        }
        SG_FREE(lm);
    }
    SG_FREE(pg);
    SG_FREE(w);
    SG_FREE(d);
    SG_FREE(gp);
    SG_FREE(g);
    SG_FREE(xp);

    return ret;
}



static int32_t line_search_backtracking(
    int32_t n,
    float64_t *x,
    float64_t *f,
    float64_t *g,
    float64_t *s,
    float64_t *stp,
    const float64_t* xp,
    const float64_t* gp,
    float64_t *wp,
    callback_data_t *cd,
    const lbfgs_parameter_t *param
    )
{
    int32_t count = 0;
    float64_t width, dg;
    float64_t finit, dginit = 0., dgtest;
    const float64_t dec = 0.5, inc = 2.1;

    /* Check the input parameters for errors. */
    if (*stp <= 0.) {
        return LBFGSERR_INVALIDPARAMETERS;
    }

    /* Compute the initial gradient in the search direction. */
	dginit = SGVector<float64_t>::dot(g, s, n);

    /* Make sure that s points to a descent direction. */
    if (0 < dginit) {
        return LBFGSERR_INCREASEGRADIENT;
    }

    /* The initial value of the objective function. */
    finit = *f;
    dgtest = param->ftol * dginit;

    for (;;) {
		std::copy(xp,xp+n,x);
		SGVector<float64_t>::add(x, 1, x, *stp, s, n);

        /* Evaluate the function and gradient values. */
        *f = cd->proc_evaluate(cd->instance, x, g, cd->n, *stp);

        ++count;

        if (*f > finit + *stp * dgtest) {
            width = dec;
        } else {
            /* The sufficient decrease condition (Armijo condition). */
            if (param->linesearch == LBFGS_LINESEARCH_BACKTRACKING_ARMIJO) {
                /* Exit with the Armijo condition. */
                return count;
	        }

	        /* Check the Wolfe condition. */
			dg = SGVector<float64_t>::dot(g, s, n);
	        if (dg < param->wolfe * dginit) {
		    width = inc;
	        } else {
		        if(param->linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE) {
		            /* Exit with the regular Wolfe condition. */
		            return count;
		        }

		        /* Check the strong Wolfe condition. */
		        if(dg > -param->wolfe * dginit) {
		            width = dec;
		        } else {
		            /* Exit with the strong Wolfe condition. */
		            return count;
		        }
            }
        }

        if (*stp < param->min_step) {
            /* The step is the minimum value. */
            return LBFGSERR_MINIMUMSTEP;
        }
        if (*stp > param->max_step) {
            /* The step is the maximum value. */
            return LBFGSERR_MAXIMUMSTEP;
        }
        if (param->max_linesearch <= count) {
            /* Maximum number of iteration. */
            return LBFGSERR_MAXIMUMLINESEARCH;
        }

        (*stp) *= width;
    }
}



static int32_t line_search_backtracking_owlqn(
    int32_t n,
    float64_t *x,
    float64_t *f,
    float64_t *g,
    float64_t *s,
    float64_t *stp,
    const float64_t* xp,
    const float64_t* gp,
    float64_t *wp,
    callback_data_t *cd,
    const lbfgs_parameter_t *param
    )
{
    int32_t i, count = 0;
    float64_t width = 0.5, norm = 0.;
    float64_t finit = *f, dgtest;

    /* Check the input parameters for errors. */
    if (*stp <= 0.) {
        return LBFGSERR_INVALIDPARAMETERS;
    }

    /* Choose the orthant for the new point. */
    for (i = 0;i < n;++i) {
        wp[i] = (xp[i] == 0.) ? -gp[i] : xp[i];
    }

    for (;;) {
        /* Update the current point. */
		std::copy(xp,xp+n,x);
		SGVector<float64_t>::add(x, 1, x, *stp, s, n);

        /* The current point is projected onto the orthant. */
        owlqn_project(x, wp, param->orthantwise_start, param->orthantwise_end);

        /* Evaluate the function and gradient values. */
        *f = cd->proc_evaluate(cd->instance, x, g, cd->n, *stp);

        /* Compute the L1 norm of the variables and add it to the object value. */
        norm = owlqn_x1norm(x, param->orthantwise_start, param->orthantwise_end);
        *f += norm * param->orthantwise_c;

        ++count;

        dgtest = 0.;
        for (i = 0;i < n;++i) {
            dgtest += (x[i] - xp[i]) * gp[i];
        }

        if (*f <= finit + param->ftol * dgtest) {
            /* The sufficient decrease condition. */
            return count;
        }

        if (*stp < param->min_step) {
            /* The step is the minimum value. */
            return LBFGSERR_MINIMUMSTEP;
        }
        if (*stp > param->max_step) {
            /* The step is the maximum value. */
            return LBFGSERR_MAXIMUMSTEP;
        }
        if (param->max_linesearch <= count) {
            /* Maximum number of iteration. */
            return LBFGSERR_MAXIMUMLINESEARCH;
        }

        (*stp) *= width;
    }
}



static int32_t line_search_morethuente(
    int32_t n,
    float64_t *x,
    float64_t *f,
    float64_t *g,
    float64_t *s,
    float64_t *stp,
    const float64_t* xp,
    const float64_t* gp,
    float64_t *wa,
    callback_data_t *cd,
    const lbfgs_parameter_t *param
    )
{
    int32_t count = 0;
    int32_t brackt, stage1, uinfo = 0;
    float64_t dg;
    float64_t stx, fx, dgx;
    float64_t sty, fy, dgy;
    float64_t fxm, dgxm, fym, dgym, fm, dgm;
    float64_t finit, ftest1, dginit, dgtest;
    float64_t width, prev_width;
    float64_t stmin, stmax;

    /* Check the input parameters for errors. */
    if (*stp <= 0.) {
        return LBFGSERR_INVALIDPARAMETERS;
    }

    /* Compute the initial gradient in the search direction. */
	dginit = SGVector<float64_t>::dot(g, s, n);

    /* Make sure that s points to a descent direction. */
    if (0 < dginit) {
        return LBFGSERR_INCREASEGRADIENT;
    }

    /* Initialize local variables. */
    brackt = 0;
    stage1 = 1;
    finit = *f;
    dgtest = param->ftol * dginit;
    width = param->max_step - param->min_step;
    prev_width = 2.0 * width;

    /*
        The variables stx, fx, dgx contain the values of the step,
        function, and directional derivative at the best step.
        The variables sty, fy, dgy contain the value of the step,
        function, and derivative at the other endpoint of
        the interval of uncertainty.
        The variables stp, f, dg contain the values of the step,
        function, and derivative at the current step.
    */
    stx = sty = 0.;
    fx = fy = finit;
    dgx = dgy = dginit;

    for (;;) {
        /*
            Set the minimum and maximum steps to correspond to the
            present interval of uncertainty.
         */
        if (brackt) {
            stmin = min2(stx, sty);
            stmax = max2(stx, sty);
        } else {
            stmin = stx;
            stmax = *stp + 4.0 * (*stp - stx);
        }

        /* Clip the step in the range of [stpmin, stpmax]. */
        if (*stp < param->min_step) *stp = param->min_step;
        if (param->max_step < *stp) *stp = param->max_step;

        /*
            If an unusual termination is to occur then let
            stp be the lowest point obtained so far.
         */
        if ((brackt && ((*stp <= stmin || stmax <= *stp) || param->max_linesearch <= count + 1 || uinfo != 0)) || (brackt && (stmax - stmin <= param->xtol * stmax))) {
            *stp = stx;
        }

        /*
            Compute the current value of x:
                x <- x + (*stp) * s.
         */
		std::copy(xp,xp+n,x);
		SGVector<float64_t>::add(x, 1, x, *stp, s, n);

        /* Evaluate the function and gradient values. */
        *f = cd->proc_evaluate(cd->instance, x, g, cd->n, *stp);
		dg = SGVector<float64_t>::dot(g, s, n);

        ftest1 = finit + *stp * dgtest;
        ++count;

        /* Test for errors and convergence. */
        if (brackt && ((*stp <= stmin || stmax <= *stp) || uinfo != 0)) {
            /* Rounding errors prevent further progress. */
            return LBFGSERR_ROUNDING_ERROR;
        }
        if (*stp == param->max_step && *f <= ftest1 && dg <= dgtest) {
            /* The step is the maximum value. */
            return LBFGSERR_MAXIMUMSTEP;
        }
        if (*stp == param->min_step && (ftest1 < *f || dgtest <= dg)) {
            /* The step is the minimum value. */
            return LBFGSERR_MINIMUMSTEP;
        }
        if (brackt && (stmax - stmin) <= param->xtol * stmax) {
            /* Relative width of the interval of uncertainty is at most xtol. */
            return LBFGSERR_WIDTHTOOSMALL;
        }
        if (param->max_linesearch <= count) {
            /* Maximum number of iteration. */
            return LBFGSERR_MAXIMUMLINESEARCH;
        }
        if (*f <= ftest1 && fabs(dg) <= param->gtol * (-dginit)) {
            /* The sufficient decrease condition and the directional derivative condition hold. */
            return count;
        }

        /*
            In the first stage we seek a step for which the modified
            function has a nonpositive value and nonnegative derivative.
         */
        if (stage1 && *f <= ftest1 && min2(param->ftol, param->gtol) * dginit <= dg) {
            stage1 = 0;
        }

        /*
            A modified function is used to predict the step only if
            we have not obtained a step for which the modified
            function has a nonpositive function value and nonnegative
            derivative, and if a lower function value has been
            obtained but the decrease is not sufficient.
         */
        if (stage1 && ftest1 < *f && *f <= fx) {
            /* Define the modified function and derivative values. */
            fm = *f - *stp * dgtest;
            fxm = fx - stx * dgtest;
            fym = fy - sty * dgtest;
            dgm = dg - dgtest;
            dgxm = dgx - dgtest;
            dgym = dgy - dgtest;

            /*
                Call update_trial_interval() to update the interval of
                uncertainty and to compute the new step.
             */
            uinfo = update_trial_interval(
                &stx, &fxm, &dgxm,
                &sty, &fym, &dgym,
                stp, &fm, &dgm,
                stmin, stmax, &brackt
                );

            /* Reset the function and gradient values for f. */
            fx = fxm + stx * dgtest;
            fy = fym + sty * dgtest;
            dgx = dgxm + dgtest;
            dgy = dgym + dgtest;
        } else {
            /*
                Call update_trial_interval() to update the interval of
                uncertainty and to compute the new step.
             */
            uinfo = update_trial_interval(
                &stx, &fx, &dgx,
                &sty, &fy, &dgy,
                stp, f, &dg,
                stmin, stmax, &brackt
                );
        }

        /*
            Force a sufficient decrease in the interval of uncertainty.
         */
        if (brackt) {
            if (0.66 * prev_width <= fabs(sty - stx)) {
                *stp = stx + 0.5 * (sty - stx);
            }
            prev_width = width;
            width = fabs(sty - stx);
        }
    }

    return LBFGSERR_LOGICERROR;
}



/**
 * Define the local variables for computing minimizers.
 */
#define USES_MINIMIZER \
    float64_t a, d, gamma, theta, p, q, r, s;

/**
 * Find a minimizer of an interpolated cubic function.
 *  @param  cm      The minimizer of the interpolated cubic.
 *  @param  u       The value of one point, u.
 *  @param  fu      The value of f(u).
 *  @param  du      The value of f'(u).
 *  @param  v       The value of another point, v.
 *  @param  fv      The value of f(v).
 *  @param  du      The value of f'(v).
 */
#define CUBIC_MINIMIZER(cm, u, fu, du, v, fv, dv) \
    d = (v) - (u); \
    theta = ((fu) - (fv)) * 3 / d + (du) + (dv); \
    p = fabs(theta); \
    q = fabs(du); \
    r = fabs(dv); \
    s = max3(p, q, r); \
    /* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */ \
    a = theta / s; \
    gamma = s * sqrt(a * a - ((du) / s) * ((dv) / s)); \
    if ((v) < (u)) gamma = -gamma; \
    p = gamma - (du) + theta; \
    q = gamma - (du) + gamma + (dv); \
    r = p / q; \
    (cm) = (u) + r * d;

/**
 * Find a minimizer of an interpolated cubic function.
 *  @param  cm      The minimizer of the interpolated cubic.
 *  @param  u       The value of one point, u.
 *  @param  fu      The value of f(u).
 *  @param  du      The value of f'(u).
 *  @param  v       The value of another point, v.
 *  @param  fv      The value of f(v).
 *  @param  du      The value of f'(v).
 *  @param  xmin    The maximum value.
 *  @param  xmin    The minimum value.
 */
#define CUBIC_MINIMIZER2(cm, u, fu, du, v, fv, dv, xmin, xmax) \
    d = (v) - (u); \
    theta = ((fu) - (fv)) * 3 / d + (du) + (dv); \
    p = fabs(theta); \
    q = fabs(du); \
    r = fabs(dv); \
    s = max3(p, q, r); \
    /* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */ \
    a = theta / s; \
    gamma = s * sqrt(max2(0, a * a - ((du) / s) * ((dv) / s))); \
    if ((u) < (v)) gamma = -gamma; \
    p = gamma - (dv) + theta; \
    q = gamma - (dv) + gamma + (du); \
    r = p / q; \
    if (r < 0. && gamma != 0.) { \
        (cm) = (v) - r * d; \
    } else if (a < 0) { \
        (cm) = (xmax); \
    } else { \
        (cm) = (xmin); \
    }

/**
 * Find a minimizer of an interpolated quadratic function.
 *  @param  qm      The minimizer of the interpolated quadratic.
 *  @param  u       The value of one point, u.
 *  @param  fu      The value of f(u).
 *  @param  du      The value of f'(u).
 *  @param  v       The value of another point, v.
 *  @param  fv      The value of f(v).
 */
#define QUARD_MINIMIZER(qm, u, fu, du, v, fv) \
    a = (v) - (u); \
    (qm) = (u) + (du) / (((fu) - (fv)) / a + (du)) / 2 * a;

/**
 * Find a minimizer of an interpolated quadratic function.
 *  @param  qm      The minimizer of the interpolated quadratic.
 *  @param  u       The value of one point, u.
 *  @param  du      The value of f'(u).
 *  @param  v       The value of another point, v.
 *  @param  dv      The value of f'(v).
 */
#define QUARD_MINIMIZER2(qm, u, du, v, dv) \
    a = (u) - (v); \
    (qm) = (v) + (dv) / ((dv) - (du)) * a;

#define fsigndiff(x, y) (*(x) * (*(y) / fabs(*(y))) < 0.)

/**
 * Update a safeguarded trial value and interval for line search.
 *
 *  The parameter x represents the step with the least function value.
 *  The parameter t represents the current step. This function assumes
 *  that the derivative at the point of x in the direction of the step.
 *  If the bracket is set to true, the minimizer has been bracketed in
 *  an interval of uncertainty with endpoints between x and y.
 *
 *  @param  x       The pointer to the value of one endpoint.
 *  @param  fx      The pointer to the value of f(x).
 *  @param  dx      The pointer to the value of f'(x).
 *  @param  y       The pointer to the value of another endpoint.
 *  @param  fy      The pointer to the value of f(y).
 *  @param  dy      The pointer to the value of f'(y).
 *  @param  t       The pointer to the value of the trial value, t.
 *  @param  ft      The pointer to the value of f(t).
 *  @param  dt      The pointer to the value of f'(t).
 *  @param  tmin    The minimum value for the trial value, t.
 *  @param  tmax    The maximum value for the trial value, t.
 *  @param  brackt  The pointer to the predicate if the trial value is
 *                  bracketed.
 *  @retval int32_t     Status value. Zero indicates a normal termination.
 *
 *  @see
 *      Jorge J. More and David J. Thuente. Line search algorithm with
 *      guaranteed sufficient decrease. ACM Transactions on Mathematical
 *      Software (TOMS), Vol 20, No 3, pp. 286-307, 1994.
 */
static int32_t update_trial_interval(
    float64_t *x,
    float64_t *fx,
    float64_t *dx,
    float64_t *y,
    float64_t *fy,
    float64_t *dy,
    float64_t *t,
    float64_t *ft,
    float64_t *dt,
    const float64_t tmin,
    const float64_t tmax,
    int32_t *brackt
    )
{
    int32_t bound;
    int32_t dsign = fsigndiff(dt, dx);
    float64_t mc; /* minimizer of an interpolated cubic. */
    float64_t mq; /* minimizer of an interpolated quadratic. */
    float64_t newt;   /* new trial value. */
    USES_MINIMIZER;     /* for CUBIC_MINIMIZER and QUARD_MINIMIZER. */

    /* Check the input parameters for errors. */
    if (*brackt) {
        if (*t <= min2(*x, *y) || max2(*x, *y) <= *t) {
            /* The trival value t is out of the interval. */
            return LBFGSERR_OUTOFINTERVAL;
        }
        if (0. <= *dx * (*t - *x)) {
            /* The function must decrease from x. */
            return LBFGSERR_INCREASEGRADIENT;
        }
        if (tmax < tmin) {
            /* Incorrect tmin and tmax specified. */
            return LBFGSERR_INCORRECT_TMINMAX;
        }
    }

    /*
        Trial value selection.
     */
    if (*fx < *ft) {
        /*
            Case 1: a higher function value.
            The minimum is brackt. If the cubic minimizer is closer
            to x than the quadratic one, the cubic one is taken, else
            the average of the minimizers is taken.
         */
        *brackt = 1;
        bound = 1;
        CUBIC_MINIMIZER(mc, *x, *fx, *dx, *t, *ft, *dt);
        QUARD_MINIMIZER(mq, *x, *fx, *dx, *t, *ft);
        if (fabs(mc - *x) < fabs(mq - *x)) {
            newt = mc;
        } else {
            newt = mc + 0.5 * (mq - mc);
        }
    } else if (dsign) {
        /*
            Case 2: a lower function value and derivatives of
            opposite sign. The minimum is brackt. If the cubic
            minimizer is closer to x than the quadratic (secant) one,
            the cubic one is taken, else the quadratic one is taken.
         */
        *brackt = 1;
        bound = 0;
        CUBIC_MINIMIZER(mc, *x, *fx, *dx, *t, *ft, *dt);
        QUARD_MINIMIZER2(mq, *x, *dx, *t, *dt);
        if (fabs(mc - *t) > fabs(mq - *t)) {
            newt = mc;
        } else {
            newt = mq;
        }
    } else if (fabs(*dt) < fabs(*dx)) {
        /*
            Case 3: a lower function value, derivatives of the
            same sign, and the magnitude of the derivative decreases.
            The cubic minimizer is only used if the cubic tends to
            infinity in the direction of the minimizer or if the minimum
            of the cubic is beyond t. Otherwise the cubic minimizer is
            defined to be either tmin or tmax. The quadratic (secant)
            minimizer is also computed and if the minimum is brackt
            then the the minimizer closest to x is taken, else the one
            farthest away is taken.
         */
        bound = 1;
        CUBIC_MINIMIZER2(mc, *x, *fx, *dx, *t, *ft, *dt, tmin, tmax);
        QUARD_MINIMIZER2(mq, *x, *dx, *t, *dt);
        if (*brackt) {
            if (fabs(*t - mc) < fabs(*t - mq)) {
                newt = mc;
            } else {
                newt = mq;
            }
        } else {
            if (fabs(*t - mc) > fabs(*t - mq)) {
                newt = mc;
            } else {
                newt = mq;
            }
        }
    } else {
        /*
            Case 4: a lower function value, derivatives of the
            same sign, and the magnitude of the derivative does
            not decrease. If the minimum is not brackt, the step
            is either tmin or tmax, else the cubic minimizer is taken.
         */
        bound = 0;
        if (*brackt) {
            CUBIC_MINIMIZER(newt, *t, *ft, *dt, *y, *fy, *dy);
        } else if (*x < *t) {
            newt = tmax;
        } else {
            newt = tmin;
        }
    }

    /*
        Update the interval of uncertainty. This update does not
        depend on the new step or the case analysis above.

        - Case a: if f(x) < f(t),
            x <- x, y <- t.
        - Case b: if f(t) <= f(x) && f'(t)*f'(x) > 0,
            x <- t, y <- y.
        - Case c: if f(t) <= f(x) && f'(t)*f'(x) < 0,
            x <- t, y <- x.
     */
    if (*fx < *ft) {
        /* Case a */
        *y = *t;
        *fy = *ft;
        *dy = *dt;
    } else {
        /* Case c */
        if (dsign) {
            *y = *x;
            *fy = *fx;
            *dy = *dx;
        }
        /* Cases b and c */
        *x = *t;
        *fx = *ft;
        *dx = *dt;
    }

    /* Clip the new trial value in [tmin, tmax]. */
    if (tmax < newt) newt = tmax;
    if (newt < tmin) newt = tmin;

    /*
        Redefine the new trial value if it is close to the upper bound
        of the interval.
     */
    if (*brackt && bound) {
        mq = *x + 0.66 * (*y - *x);
        if (*x < *y) {
            if (mq < newt) newt = mq;
        } else {
            if (newt < mq) newt = mq;
        }
    }

    /* Return the new trial value. */
    *t = newt;
    return 0;
}





static float64_t owlqn_x1norm(
    const float64_t* x,
    const int32_t start,
    const int32_t n
    )
{
    int32_t i;
    float64_t norm = 0.;

    for (i = start;i < n;++i) {
        norm += fabs(x[i]);
    }

    return norm;
}

static void owlqn_pseudo_gradient(
    float64_t* pg,
    const float64_t* x,
    const float64_t* g,
    const int32_t n,
    const float64_t c,
    const int32_t start,
    const int32_t end
    )
{
    int32_t i;

    /* Compute the negative of gradients. */
    for (i = 0;i < start;++i) {
        pg[i] = g[i];
    }

    /* Compute the psuedo-gradients. */
    for (i = start;i < end;++i) {
        if (x[i] < 0.) {
            /* Differentiable. */
            pg[i] = g[i] - c;
        } else if (0. < x[i]) {
            /* Differentiable. */
            pg[i] = g[i] + c;
        } else {
            if (g[i] < -c) {
                /* Take the right partial derivative. */
                pg[i] = g[i] + c;
            } else if (c < g[i]) {
                /* Take the left partial derivative. */
                pg[i] = g[i] - c;
            } else {
                pg[i] = 0.;
            }
        }
    }

    for (i = end;i < n;++i) {
        pg[i] = g[i];
    }
}

static void owlqn_project(
    float64_t* d,
    const float64_t* sign,
    const int32_t start,
    const int32_t end
    )
{
    int32_t i;

    for (i = start;i < end;++i) {
        if (d[i] * sign[i] <= 0) {
            d[i] = 0;
        }
    }
}

}
