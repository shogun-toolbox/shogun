/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Purpose:     solves quadratic programming problem for pattern recognition
 *              for support vectors
 *
 * Written (W) 1997-1998 Alex J. Smola
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1997-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/config.h>
#ifdef USE_GPL_SHOGUN

namespace shogun
{
/* verbosity levels */

#define QUIET 0
#define STATUS 1
#define FLOOD 2

/* status outputs */

#define STILL_RUNNING               0
#define OPTIMAL_SOLUTION            1
#define SUBOPTIMAL_SOLUTION         2
#define ITERATION_LIMIT             3
#define PRIMAL_INFEASIBLE           4
#define DUAL_INFEASIBLE             5
#define PRIMAL_AND_DUAL_INFEASIBLE  6
#define INCONSISTENT                7
#define PRIMAL_UNBOUNDED            8
#define DUAL_UNBOUNDED              9
#define TIME_LIMIT                  10

/*
 * solve the quadratic programming problem
 *
 * minimize   c' * x + 1/2 x' * H * x
 * subject to A*x = b
 *            l <= x <= u
 *
 *  for a documentation see R. Vanderbei, LOQO: an Interior Point Code
 *                          for Quadratic Programming
 */

/**
 * n   : number of primal variables
 * m   : number of constraints (typically 1)
 * h_x : dot product matrix (n.n)
 * a   : constraint matrix (n.m)
 * b   : constant term (m)
 * l   : lower bound (n)
 * u   : upper bound (m)
 *
 * primal : workspace for primal variables, has to be of size 3 n
 *
 *  x = primal;			n
 *  g = x + n;			n
 *  t = g + n;			n
 *
 * dual : workspace for dual variables, has to be of size m + 2 n
 *
 *  y = dual;			m
 *  z = y + m;			n
 *  s = z + n;			n
 *
 * verb       : verbosity level
 * sigfig_max : number of significant digits
 * counter_max: stopping criterion
 * restart    : 1 if restart desired
 *
 */
int32_t pr_loqo(
	int32_t n, int32_t m, float64_t c[], float64_t h_x[], float64_t a[],
	float64_t b[], float64_t l[], float64_t u[], float64_t primal[],
	float64_t dual[], int32_t verb, float64_t sigfig_max, int32_t counter_max,
	float64_t margin, float64_t bound, int32_t restart);
}
#endif //USE_GPL_SHOGUN
