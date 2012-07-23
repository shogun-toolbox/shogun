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

#ifndef  SFA_SLEP
#define  SFA_SLEP

/* 
   Revision History

   First Version available on October 10, 2009 

   A runnable version on October 15, 2009

   Major revision on October 29, 2009
   (Some functions appearing in a previous version have deleted, please refer to the previous version for the old functions.
   Some new functions have been added as well)

*/

/*

   Files contained in this header file sfa.h:

   1. Algorithms for solving the linear system A A^T z0 = Av (see the description of A from the following context)

   void Thomas(double *zMax, double *z0, 
   double * Av, int nn)

   void Rose(double *zMax, double *z0, 
   double * Av, int nn)

   int supportSet(double *x, double *v, double *z, 
   double *g, int * S, double lambda, int nn)

   void dualityGap(double *gap, double *z, 
   double *g, double *s, double *Av, 
   double lambda, int nn)

   void dualityGap2(double *gap, double *z, 
   double *g, double *s, double *Av, 
   double lambda, int nn)


   2. The Subgraident Finding Algorithm (SFA) for solving problem (4) (refer to the description of the problem for detail) 

   int sfa(double *x,     double *gap,
   double *z,     double *z0,   double * v,   double * Av, 
   double lambda, int nn,       int maxStep,
   double *s,     double *g,
   double tol,    int tau,       int flag)

   int sfa_special(double *x,     double *gap,
   double *z,     double * v,   double * Av, 
   double lambda, int nn,       int maxStep,
   double *s,     double *g,
   double tol,    int tau)

   int sfa_one(double *x,     double *gap,
   double *z,     double * v,   double * Av, 
   double lambda, int nn,       int maxStep,
   double *s,     double *g,
   double tol,    int tau)


*/


/*

   Some mathematical background.

   In this file, we discuss how to solve the following subproblem,

   min_x  1/2 \|x-v\|^2  + lambda \|A x\|_1,                 (1)

   which is a key problem used in the Fused Lasso Signal Approximator (FLSA).

   Also, note that, FLSA is a building block for solving the optimation problmes with fused Lasso penalty.

   In (1), x and v are n-dimensional vectors, 
   and A is a matrix with size (n-1) x n, and is defined as follows (e.g., n=4):
   A= [ -1  1  0  0;
   0  -1 1  0;
   0  0  -1 1]

   The above problem can be reformulated as the following equivalent min-max optimization problem

   min_x  max_z  1/2 \|x-v\|^2  + <A x, z>
   subject to   \|z\|_{infty} \leq lambda                     (2)


   It is easy to get that, at the optimal point

   x = v - AT z,                             (3)

   where z is the optimal solution to the following optimization problem

   min_z  1/2  z^T A AT z - < z, A v>,
   subject to  \|z\|_{infty} \leq lambda                      (4)



   Let B=A A^T. It is easy to get that B is a (n-1) x (n-1) tridiagonal matrix.
   When n=5, B is defined as:
   B= [ 2  -1   0    0;
   -1  2   -1   0;
   0  -1   2    -1;
   0   0   -1   2]

   Let z0 be the solution to the linear system:

   A A^T * z0 = A * v                  (5)

   The problem (5) can be solve by the Thomas Algorithm, in about 5n multiplications and 4n additions.

   It can also be solved by the Rose's Algorithm, in about 2n multiplications and 2n additions.

   Moreover, considering the special structure of the matrix A (and B), 
   it can be solved in about n multiplications and 3n additions

   If lambda \geq \|z0\|_{infty}, x_i= mean(v), for all i, 
   the problem (1) admits near analytical solution


   We have also added the restart technique, please refer to our paper for detail!

*/


void Thomas(double *zMax, double *z0, double * Av, int nn);

void Rose(double *zMax, double *z0,	double * Av, int nn);

/*
////////////////    compute x for restarting \\\\\\\\\\\\\\\\\\\\\\\\\

x=omega(z)

v: the vector to be projected
z: the approximate solution
g: the gradient at z (g should be computed before calling this function

nn: the length of z, g, and S (maximal length for S)

n:  the length of x and v

S: records the indices of the elements in the support set
*/
int supportSet(double *x, double *v, double *z, double *g, int * S, double lambda, int nn);

/*
////////////  Computing the duality gap \\\\\\\\\\\\\\\\\\\\\\\\\\

we compute the duality corresponding the solution z

z: the approximate solution
g: the gradient at z (we recompute the gradient)
s: an auxiliary variable
Av: A*v

nn: the lenght for z, g, s, and Av

The variables g and s shall be revised.

The variables z and Av remain unchanged.
*/
void dualityGap(double *gap, double *z, double *g, double *s, double *Av, double lambda, int nn);

/*
   Similar to dualityGap,

   The difference is that, we assume that g has been computed.
   */
void dualityGap2(double *gap, double *z, double *g, double *s, double *Av, double lambda, int nn);

/*
generateSolution:

generate the solution x based on the information of z and g 
(!!!!we assume that g has been computed as the gradient of z!!!!)

*/
int generateSolution(double *x, double *z, double *gap,
		double *v, double *Av,
		double *g, double *s, int *S,
		double lambda, int nn);

void restartMapping(double *g, double *z,  double * v, 
		double lambda, int nn);

/*
/////////////////////////////////////// Explanation for the function sfa \\\\\\\\\\\\\\\\\\\\\\\\\\\\

Our objective is to solve the fused Lasso signal approximator (flsa) problem:

min_x  g(x) 1/2 \|x-v\|^2  + lambda \|A x\|_1,                      (1)

Let x* be the solution (which is unique), it satisfies

0 in  x* - v +  A^T * lambda *SGN(Ax*)                     (2)

To solve x*, it suffices to find

y*  in A^T * lambda *SGN(Ax*)                              (3)
that satisfies

x* - v + y* =0                                             (4)
which leads to
x*= v - y*                                                 (5)

Due to the uniqueness of x*, we conclude that y* is unique. 

As y* is a subgradient of lambda \|A x*\|_1, 
we name our method as Subgradient Finding Algorithm (sfa).

y* in (3) can be further written as

y*= A^T * z*                                               (6)
where

z* in lambda* SGN (Ax*)                                    (7)

From (6), we have
z* = (A A^T)^{-1} A * y*                                   (8)

Therefore, from the uqniueness of y*, we conclude that z* is also unique.
Next, we discuss how to solve this unique z*.

The problem (1) can reformulated as the following equivalent problem:	 

min_x  max_z  f(x, z)= 1/2 \|x-v\|^2  + <A x, z>
subject to   \|z\|_{infty} \leq lambda                                  (9)

At the saddle point, we have

x = v - AT z,                                            (10)

which somehow concides with (5) and (6)

Plugging (10) into (9), we obtain the problem

min_z  1/2  z^T A AT z - < z, A v>,
subject to  \|z\|_{infty} \leq lambda,                             (11)

In this program, we apply the Nesterov's method for solving (11).


Duality gap:

At a given point z0, we compute x0= v - A^T z0.
It is easy to show that
min_x f(x, z0) = f(x0, z0) <= max_z f(x0, z)               (12)

Moreover, we have
max_z f(x0, z) - min_x f(x, z0) 
<= lambda * \|A x0\|_1 - < z0, Av - A A^T z0>           (13)

It is also to get that

f(x0, z0) <= f(x*, z*) <= max_z f(x0, z)                   (14)

g(x*)=f(x*, z*)                                            (15)

g(x0)=max_z f(x0, z)                                       (17)

	Therefore, we have

g(x0)-g(x*) <= lambda * \|A x0\|_1 - < z0, Av - A A^T z0>  (18)


	We have applied a restarting technique, which is quite involved; and thus, we do not explain here.

	/////////////////////////////////////// Explanation for the function sfa \\\\\\\\\\\\\\\\\\\\\\\\\\\\
		*/


		/*
		////////////               sfa              \\\\\\\\\\\\\\\\\\\\\

		For sfa, the stepsize of the Nesterov's method is fixed to 1/4, so that no line search is needed.



		Explanation of the parameters:

		Output parameters
		x:    the solution to the primal problem
		gap:  the duality gap (pointer)

		Input parameters
		z:    the solution to the dual problem (before calling this function, z contains a starting point)
		!!!!we assume that the starting point has been successfully initialized in z !!!!
		z0:   a variable used for multiple purposes:
		1) the previous solution z0
		2) the difference between z and z0, i.e., z0=z- z0

		lambda:   the regularization parameter (and the radius of the infity ball, see (11)).
		nn:       the length of z, z0, Av, g, and s
		maxStep:  the maximal number of iterations

		v:    the point to be projected (not changed after the program)
		Av:   A*v (not changed after the program)

		s:        the search point (used for multiple purposes)
		g:        the gradient at g (and it is also used for multiple purposes)

		tol:      the tolerance of the gap
		tau:  the duality gap or the restarting technique is done every tau steps
		flag: if flag=1,  we apply the resart technique
		flag=2,  just run the SFA algorithm, terminate it when the absolution change is less than tol
		flag=3,  just run the SFA algorithm, terminate it when the duality gap is less than tol
		flag=4,  just run the SFA algorithm, terminate it when the relative duality gap is less than tol


		We would like to emphasis that the following assumptions 
		have been checked in the functions that call this function:
		1) 0< lambda < z_max
		2) nn >=2
		3) z has been initialized with a starting point
		4) z0 has been initialized with all zeros

		The termination condition is checked every tau iterations.

		For the duality gap, please refer to (12-18)
		*/
int sfa(double *x,     double *gap, int * activeS,
		double *z,     double *z0,   double * v,   double * Av, 
		double lambda, int nn,       int maxStep,
		double *s,     double *g,
		double tol,    int tau,       int flag);

/*

   Refer to sfa for the defintions of the variables  

   In this file, we restart the program every step, and neglect the gradient step.

   It seems that, this program does not converge.

   This function shows that the gradient step is necessary.
   */
int sfa_special(double *x,     double *gap,  int * activeS,
		double *z,     double * v,   double * Av, 
		double lambda, int nn,       int maxStep,
		double *s,     double *g,
		double tol,    int tau);

/*
   We do one gradient descent, and then restart the program
   */
int sfa_one(double *x,     double *gap, int * activeS,
		double *z,     double * v,   double * Av, 
		double lambda, int nn,       int maxStep,
		double *s,     double *g,
		double tol,    int tau);
#endif   /* ----- #ifndef SFA_SLEP  ----- */

