/*   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *   Copyright (C) 2009 - 2012 Jun Liu and Jieping Ye 
 */

#ifndef  FLSA_SLEP
#define  FLSA_SLEP

/*

   In this file, we solve the Fused Lasso Signal Approximator (FLSA) problem:

   min_x  1/2 \|x-v\|^2  + lambda1 * \|x\|_1 + lambda2 * \|A x\|_1,      (1)

   It can be shown that, if x* is the solution to

   min_x  1/2 \|x-v\|^2  + lambda2 \|A x\|_1,                            (2)

   then 
   x**= sgn(x*) max(|x*|-lambda_1, 0)                                    (3)

   is the solution to (1).

   By some derivation (see the description in sfa.h), (2) can be solved by

   x*= v - A^T z*,

   where z* is the optimal solution to

   min_z  1/2  z^T A AT z - < z, A v>,
   subject to  \|z\|_{infty} \leq lambda2                             (4)
   */



/*

   In flsa, we solve (1) corresponding to a given (lambda1, lambda2)

   void flsa(double *x, double *z, double *gap,
   double * v, double *z0, 
   double lambda1, double lambda2, int n, 
   int maxStep, double tol, int flag)

   Output parameters:
x:        the solution to problem (1)
z:        the solution to problem (4)
infor:    the information about running the subgradient finding algorithm
infor[0] = gap:         the computed gap (either the duality gap
or the summation of the absolute change of the adjacent solutions)
infor[1] = steps:       the number of iterations
infor[2] = lambad2_max: the maximal value of lambda2_max
infor[3] = numS:        the number of elements in the support set

Input parameters:
v:        the input vector to be projected
z0:       a guess of the solution of z

lambad1:  the regularization parameter
labmda2:  the regularization parameter
n:        the length of v and x

maxStep:  the maximal allowed iteration steps
tol:      the tolerance parameter
tau:      the program sfa is checked every tau iterations for termination
flag:     the flag for initialization and deciding calling sfa
switch ( flag )
1-4, 11-14: sfa

switch ( flag )
case 1, 2, 3, or 4: 
z0 is a "good" starting point 
(such as the warm-start of the previous solution,
or the user want to test the performance of this starting point;
the starting point shall be further projected to the L_{infty} ball,
to make sure that it is feasible)

case 11, 12, 13, or 14: z0 is a "random" guess, and thus not used
(we shall initialize z as follows:
if lambda2 >= 0.5 * lambda_2^max, we initialize the solution of the linear system;
if lambda2 <  0.5 * lambda_2^max, we initialize with zero
this solution is projected to the L_{infty} ball)

switch( flag )
5, 15: sfa_special

switch( flag )
5:  z0 is a good starting point
15: z0 is a bad starting point, use the solution of the linear system


switch( flag )
6, 16: sfa_one

switch( flag )
6:  z0 is a good starting point
16: z0 is a bad starting point, use the solution of the linear system

Revision made on October 31, 2009.
The input variable z0 is not modified after calling sfa. For this sake, we allocate a new variable zz to replace z0.
*/
void flsa(double *x, double *z, double *infor,
		double * v, double *z0, 
		double lambda1, double lambda2, int n, 
		int maxStep, double tol, int tau, int flag);
#endif   /* ----- #ifndef FLSA_SLEP  ----- */
