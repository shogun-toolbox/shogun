/*
 * File:        pr_loqo.h
 * Purpose:     solves quadratic programming problem for pattern recognition
 *              for support vectors
 *
 * Author:      Alex J. Smola
 * Created:     10/14/97
 * Updated:     11/08/97
 *
 * 
 * Copyright (c) 1997  GMD Berlin - All rights reserved
 * THIS IS UNPUBLISHED PROPRIETARY SOURCE CODE of GMD Berlin
 * The copyright notice above does not evidence any
 * actual or intended publication of this work.
 *
 * Unauthorized commercial use of this software is not allowed
 */

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

/*
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

int pr_loqo(int n, int m, double c[], double h_x[], double a[], double b[],
	    double l[], double u[], double primal[], double dual[], 
	    int verb, double sigfig_max, int counter_max, 
	    double margin, double bound, int restart);

/* 
 * compile with
 cc -O4 -c pr_loqo.c
 cc -xO4 -fast -xarch=v8plus -xchip=ultra -xparallel -c pr_loqo.c
 mex pr_loqo_c.c pr_loqo.o
 cmex4 pr_loqo_c.c pr_loqo.o -DMATLAB4 -o pr_loqo_c4
 *
 */




