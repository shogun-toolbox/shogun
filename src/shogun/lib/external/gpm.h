/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 *** Authors:                                                               ***
 ***  Thomas Serafini, Luca Zanni                                           ***
 ***   Dept. of Mathematics, University of Modena and Reggio Emilia - ITALY ***
 ***   serafini.thomas@unimo.it, zanni.luca@unimo.it                        ***
 ***  Gaetano Zanghirati                                                    ***
 ***   Dept. of Mathematics, University of Ferrara - ITALY                  ***
 ***   g.zanghirati@unife.it                                                ***
 ***                                                                        ***
 *** Software homepage: http://dm.unife.it/gpdt                             ***
 ***                                                                        ***
 *** This work is supported by the Italian FIRB Projects                    ***
 ***      'Statistical Learning: Theory, Algorithms and Applications'       ***
 ***      (grant RBAU01877P), http://slipguru.disi.unige.it/ASTA            ***
 *** and                                                                    ***
 ***      'Parallel Algorithms and Numerical Nonlinear Optimization'        ***
 ***      (grant RBAU01JYPN), http://dm.unife.it/pn2o                       ***
 ***                                                                        ***
 *** Copyright (C) 2004-2008 by T. Serafini, G. Zanghirati, L. Zanni.       ***
 ***                                                                        ***
 *** SHOGUN adaptions  Written (W) 2006-2008 Soeren Sonnenburg              ***
 */

#include <lib/common.h>

namespace shogun
{
/** gpm solver
 * @param Solver
 * @param Projector
 * @param n
 * @param A
 * @param b
 * @param c
 * @param e
 * @param iy
 * @param x
 * @param tol
 * @param ls
 * @param proj
 */
int32_t gpm_solver(
	int32_t Solver, int32_t Projector, int32_t n, float32_t *A, float64_t *b,
	float64_t c, float64_t e, int32_t *iy, float64_t *x, float64_t tol,
	int32_t *ls = 0, int32_t *proj = 0);
}
