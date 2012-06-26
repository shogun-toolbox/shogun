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

#ifndef EPPHQ1_SLEP
#define EPPHQ1_SLEP

/* -------------------------- Function eplb -----------------------------

   Euclidean Projection onto l1 Ball (eplb)

   min  1/2 ||x- v||_2^2
   s.t. ||x||_1 <= z

   which is converted to the following zero finding problem

   f(lambda)= sum( max( |v|-lambda,0) )-z=0

   For detail, please refer to our paper:

   Jun Liu and Jieping Ye. Efficient Euclidean Projections in Linear Time,
   ICML 2009.  

   Usage (in matlab):
   [x, lambda, iter_step]=eplb(v, n, z, lambda0);

   -------------------------- Function eplb -----------------------------
   */
void eplb(double * x, double *root, int * steps, double * v,int n, double z, double lambda0);

/* -------------------------- Function epp1 -----------------------------

   The L1-norm Regularized Euclidean Projection (epp1)

   min  1/2 ||x- v||_2^2 + rho ||x||_1

   which has the closed form solution

   x= sign(v) max( |v|- rho, 0)

   Usage (in matlab)
   x=epp1(v, n, rho); 

   -------------------------- Function epp1 -----------------------------
   */
void  epp1(double *x, double *v, int n, double rho);

/* -------------------------- Function epp2 -----------------------------

   The L2-norm Regularized Euclidean Projection (epp2)

   min  1/2 ||x- v||_2^2 + rho ||x||_2

   which has the closed form solution

   x= max( ||v||_2- rho, 0) / ||v||_2 * v

   Usage (in matlab)
   x=epp2(v, n, rho); 

   -------------------------- Function epp2 -----------------------------
   */
void  epp2(double *x, double *v, int n, double rho);

/* -------------------------- Function eppInf -----------------------------

   The LInf-norm Regularized Euclidean Projection (eppInf)

   min  1/2 ||x- v||_2^2 + rho ||x||_Inf

   which is can be solved by using eplb

   Usage (in matlab)
   [x, lambda, iter_step]=eppInf(v, n, rho, rho0); 

   -------------------------- Function eppInf -----------------------------
   */
void  eppInf(double *x, double * c, int * iter_step, double *v,  int n, double rho, double c0);

/* -------------------------- Function zerofind -----------------------------
 
   Find the root for the function: f(x) = x + c x^{p-1} - v, 
   0 <= x <= v, v>=0
   1< p < infty, p \neq 2

   Property: when p>2, f(x) is a convex function
   when 1<p<2, f(x) is a concave function

   Method: we use Newton's method (other methods such as bisection can also work)

   Note: we donot check the valid of the parameter. 
   Since it is only employed in eepO, 
   we can assure that these parameters satisfy the above conditions.

   Usage (in matlab)
   [root, interStep]=eppInf(v, p, c, x0); 

   -------------------------- Function zerofind -----------------------------
   */
void zerofind(double *root, int * iterStep, double v, double p, double c, double x0);

/* -------------------------- Function norm -----------------------------

   Compute the p-norm

   -------------------------- Function norm -----------------------------
   */
double norm(double * v, double p, int n);

/* -------------------------- Function eppInf -----------------------------

   The Lp-norm Regularized Euclidean Projection (eppO) for 1< p<Inf

   min  1/2 ||x- v||_2^2 + rho ||x||_p

   We solve two simple zero finding algorithms

   Usage (in matlab)
   [x, c, iter_step]=eppO(v, n, rho, p); 

   -------------------------- Function eppInf -----------------------------
   */
void  eppO(double *x, double * cc, int * iter_step, double *v,  int n, double rho, double p);

/* -------------------------- Function epp -----------------------------

   The Lp-norm Regularized Euclidean Projection (epp) for all p>=1

   min  1/2 ||x- v||_2^2 + rho ||x||_p

   This function uses the previously defined functions.

   Usage (in matlab)
   [x, c, iter_step]=eppO(v, n, rho, p, c0); 

   -------------------------- Function epp -----------------------------
   */
void epp(double *x, double * c, int * iter_step, double * v, int n, double rho, double p, double c0);
#endif   /* ----- #ifndef EPPHQ1_SLEP  ----- */

