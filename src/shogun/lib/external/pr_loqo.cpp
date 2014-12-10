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

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/external/pr_loqo.h>

#include <shogun/mathematics/Math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

namespace shogun
{

#define PREDICTOR 1
#define CORRECTOR 2

/*****************************************************************
  replace this by any other function that will exit gracefully
  in a larger system
  ***************************************************************/

void nrerror(char error_text[])
{
	SG_SDEBUG("terminating optimizer - %s\n", error_text)
 // exit(1);
}

/*****************************************************************
   taken from numerical recipes and modified to accept pointers
   moreover numerical recipes code seems to be buggy (at least the
   ones on the web)

   cholesky solver and backsubstitution
   leaves upper right triangle intact (rows first order)
   ***************************************************************/

#ifdef HAVE_LAPACK
bool choldc(float64_t* a, int32_t n, float64_t* p)
{
	if (n<=0)
		return false;

	float64_t* a2=SG_MALLOC(float64_t, n*n);

	for (int32_t i=0; i<n; i++)
	{
		for (int32_t j=0; j<n; j++)
			a2[n*i+j]=a[n*i+j];
	}

	/* int for calling external lib */
	int result=clapack_dpotrf(CblasRowMajor, CblasUpper, (int) n, a2, (int) n);

	for (int32_t i=0; i<n; i++)
		p[i]=a2[(n+1)*i];

	for (int32_t i=0; i<n; i++)
	{
		for (int32_t j=i+1; j<n; j++)
		{
			a[n*j+i]=a2[n*i+j];
		}
	}

	if (result>0)
		SG_SDEBUG("Choldc failed, matrix not positive definite\n")

	SG_FREE(a2);

	return result==0;
}
#else
bool choldc(float64_t a[], int32_t n, float64_t p[])
{
	void nrerror(char error_text[]);
	int32_t i, j, k;
	float64_t sum;

	for (i = 0; i < n; i++)
	{
		for (j = i; j < n; j++)
		{
			sum=a[n*i + j];

			for (k=i-1; k>=0; k--)
				sum -= a[n*i + k]*a[n*j + k];

			if (i == j)
			{
				if (sum <= 0.0)
				{
					SG_SDEBUG("Choldc failed, matrix not positive definite")
					sum = 0.0;
					return false;
				}

				p[i]=sqrt(sum);

			}
			else
				a[n*j + i] = sum/p[i];
		}
	}

	return true;
}
#endif

void cholsb(
	float64_t a[], int32_t n, float64_t p[], float64_t b[], float64_t x[])
{
  int32_t i, k;
  float64_t sum;

  for (i=0; i<n; i++) {
    sum=b[i];
    for (k=i-1; k>=0; k--) sum -= a[n*i + k]*x[k];
    x[i]=sum/p[i];
  }

  for (i=n-1; i>=0; i--) {
    sum=x[i];
    for (k=i+1; k<n; k++) sum -= a[n*k + i]*x[k];
    x[i]=sum/p[i];
  }
}

/*****************************************************************
  sometimes we only need the forward or backward pass of the
  backsubstitution, hence we provide these two routines separately
  ***************************************************************/

void chol_forward(
	float64_t a[], int32_t n, float64_t p[], float64_t b[], float64_t x[])
{
  int32_t i, k;
  float64_t sum;

  for (i=0; i<n; i++) {
    sum=b[i];
    for (k=i-1; k>=0; k--) sum -= a[n*i + k]*x[k];
    x[i]=sum/p[i];
  }
}

void chol_backward(
	float64_t a[], int32_t n, float64_t p[], float64_t b[], float64_t x[])
{
  int32_t i, k;
  float64_t sum;

  for (i=n-1; i>=0; i--) {
    sum=b[i];
    for (k=i+1; k<n; k++) sum -= a[n*k + i]*x[k];
    x[i]=sum/p[i];
  }
}

/*****************************************************************
  solves the system | -H_x A' | |x_x| = |c_x|
                    |  A   H_y| |x_y|   |c_y|

  with H_x (and H_y) positive (semidefinite) matrices
  and n, m the respective sizes of H_x and H_y

  for variables see pg. 48 of notebook or do the calculations on a
  sheet of paper again

  predictor solves the whole thing, corrector assues that H_x didn't
  change and relies on the results of the predictor. therefore do
  _not_ modify workspace

  if you want to speed tune anything in the code here's the right
  place to do so: about 95% of the time is being spent in
  here. something like an iterative refinement would be nice,
  especially when switching from float64_t to single precision. if you
  have a fast parallel cholesky use it instead of the numrec
  implementations.

  side effects: changes H_y (but this is just the unit matrix or zero anyway
  in our case)
  ***************************************************************/

bool solve_reduced(
	int32_t n, int32_t m, float64_t h_x[], float64_t h_y[], float64_t a[],
	float64_t x_x[], float64_t x_y[], float64_t c_x[], float64_t c_y[],
	float64_t workspace[], int32_t step)
{
  int32_t i,j,k;

  float64_t *p_x;
  float64_t *p_y;
  float64_t *t_a;
  float64_t *t_c;
  float64_t *t_y;

  p_x = workspace;		/* together n + m + n*m + n + m = n*(m+2)+2*m */
  p_y = p_x + n;
  t_a = p_y + m;
  t_c = t_a + n*m;
  t_y = t_c + n;

  if (step == PREDICTOR) {
    if (!choldc(h_x, n, p_x))	/* do cholesky decomposition */
		return false;

    for (i=0; i<m; i++)         /* forward pass for A' */
      chol_forward(h_x, n, p_x, a+i*n, t_a+i*n);

    for (i=0; i<m; i++)         /* compute (h_y + a h_x^-1A') */
      for (j=i; j<m; j++)
	for (k=0; k<n; k++)
	  h_y[m*i + j] += t_a[n*j + k] * t_a[n*i + k];

    choldc(h_y, m, p_y);	/* and cholesky decomposition */
  }

  chol_forward(h_x, n, p_x, c_x, t_c);
				/* forward pass for c */

  for (i=0; i<m; i++) {		/* and solve for x_y */
    t_y[i] = c_y[i];
    for (j=0; j<n; j++)
      t_y[i] += t_a[i*n + j] * t_c[j];
  }

  cholsb(h_y, m, p_y, t_y, x_y);

  for (i=0; i<n; i++) {		/* finally solve for x_x */
    t_c[i] = -t_c[i];
    for (j=0; j<m; j++)
      t_c[i] += t_a[j*n + i] * x_y[j];
  }

  chol_backward(h_x, n, p_x, t_c, x_x);
  return true;
}

/*****************************************************************
  matrix vector multiplication (symmetric matrix but only one triangle
  given). computes m*x = y
  no need to tune it as it's only of O(n^2) but cholesky is of
  O(n^3). so don't waste your time _here_ although it isn't very
  elegant.
  ***************************************************************/

void matrix_vector(int32_t n, float64_t m[], float64_t x[], float64_t y[])
{
  int32_t i, j;

  for (i=0; i<n; i++) {
    y[i] = m[(n+1) * i] * x[i];

    for (j=0; j<i; j++)
      y[i] += m[i + n*j] * x[j];

    for (j=i+1; j<n; j++)
      y[i] += m[n*i + j] * x[j];
  }
}

/*****************************************************************
  call only this routine; this is the only one you're interested in
  for doing quadratical optimization

  the restart feature exists but it may not be of much use due to the
  fact that an initial setting, although close but not very close the
  the actual solution will result in very good starting diagnostics
  (primal and dual feasibility and small infeasibility gap) but incur
  later stalling of the optimizer afterwards as we have to enforce
  positivity of the slacks.
  ***************************************************************/

int32_t pr_loqo(
	int32_t n, int32_t m, float64_t c[], float64_t h_x[], float64_t a[],
	float64_t b[], float64_t l[], float64_t u[], float64_t primal[],
	float64_t dual[], int32_t verb, float64_t sigfig_max, int32_t counter_max,
	float64_t margin, float64_t bound, int32_t restart)
{
  /* the knobs to be tuned ... */
  /* float64_t margin = -0.95;	   we will go up to 95% of the
				   distance between old variables and zero */
  /* float64_t bound = 10;		   preset value for the start. small
				   values give good initial
				   feasibility but may result in slow
				   convergence afterwards: we're too
				   close to zero */
  /* to be allocated */
  float64_t *workspace;
  float64_t *diag_h_x;
  float64_t *h_y;
  float64_t *c_x;
  float64_t *c_y;
  float64_t *h_dot_x;
  float64_t *rho;
  float64_t *nu;
  float64_t *tau;
  float64_t *sigma;
  float64_t *gamma_z;
  float64_t *gamma_s;

  float64_t *hat_nu;
  float64_t *hat_tau;

  float64_t *delta_x;
  float64_t *delta_y;
  float64_t *delta_s;
  float64_t *delta_z;
  float64_t *delta_g;
  float64_t *delta_t;

  float64_t *d;

  /* from the header - pointers into primal and dual */
  float64_t *x;
  float64_t *y;
  float64_t *g;
  float64_t *z;
  float64_t *s;
  float64_t *t;

  /* auxiliary variables */
  float64_t b_plus_1;
  float64_t c_plus_1;

  float64_t x_h_x;
  float64_t primal_inf;
  float64_t dual_inf;

  float64_t sigfig;
  float64_t primal_obj, dual_obj;
  float64_t mu;
  float64_t alfa=-1;
  int32_t counter = 0;

  int32_t status = STILL_RUNNING;

  int32_t i,j;

  /* memory allocation */
  workspace = SG_MALLOC(float64_t, (n*(m+2)+2*m));
  diag_h_x  = SG_MALLOC(float64_t, n);
  h_y       = SG_MALLOC(float64_t, m*m);
  c_x       = SG_MALLOC(float64_t, n);
  c_y       = SG_MALLOC(float64_t, m);
  h_dot_x   = SG_MALLOC(float64_t, n);

  rho       = SG_MALLOC(float64_t, m);
  nu        = SG_MALLOC(float64_t, n);
  tau       = SG_MALLOC(float64_t, n);
  sigma     = SG_MALLOC(float64_t, n);

  gamma_z   = SG_MALLOC(float64_t, n);
  gamma_s   = SG_MALLOC(float64_t, n);

  hat_nu    = SG_MALLOC(float64_t, n);
  hat_tau   = SG_MALLOC(float64_t, n);

  delta_x   = SG_MALLOC(float64_t, n);
  delta_y   = SG_MALLOC(float64_t, m);
  delta_s   = SG_MALLOC(float64_t, n);
  delta_z   = SG_MALLOC(float64_t, n);
  delta_g   = SG_MALLOC(float64_t, n);
  delta_t   = SG_MALLOC(float64_t, n);

  d         = SG_MALLOC(float64_t, n);

  /* pointers into the external variables */
  x = primal;			/* n */
  g = x + n;			/* n */
  t = g + n;			/* n */

  y = dual;			/* m */
  z = y + m;			/* n */
  s = z + n;			/* n */

  /* initial settings */
  b_plus_1 = 1;
  c_plus_1 = 0;
  for (i=0; i<n; i++) c_plus_1 += c[i];

  /* get diagonal terms */
  for (i=0; i<n; i++) diag_h_x[i] = h_x[(n+1)*i];

  /* starting point */
  if (restart == 1) {
				/* x, y already preset */
    for (i=0; i<n; i++) {	/* compute g, t for primal feasibility */
      g[i] = CMath::max(CMath::abs(x[i] - l[i]), bound);
      t[i] = CMath::max(CMath::abs(u[i] - x[i]), bound);
    }

    matrix_vector(n, h_x, x, h_dot_x); /* h_dot_x = h_x * x */

    for (i=0; i<n; i++) {	/* sigma is a dummy variable to calculate z, s */
      sigma[i] = c[i] + h_dot_x[i];
      for (j=0; j<m; j++)
	sigma[i] -= a[n*j + i] * y[j];

      if (sigma[i] > 0) {
	s[i] = bound;
	z[i] = sigma[i] + bound;
      }
      else {
	s[i] = bound - sigma[i];
	z[i] = bound;
      }
    }
  }
  else {			/* use default start settings */
    for (i=0; i<m; i++)
      for (j=i; j<m; j++)
	h_y[i*m + j] = (i==j) ? 1 : 0;

    for (i=0; i<n; i++) {
      c_x[i] = c[i];
      h_x[(n+1)*i] += 1;
    }

    for (i=0; i<m; i++)
      c_y[i] = b[i];

    /* and solve the system [-H_x A'; A H_y] [x, y] = [c_x; c_y] */
    solve_reduced(n, m, h_x, h_y, a, x, y, c_x, c_y, workspace,
		  PREDICTOR);

    /* initialize the other variables */
    for (i=0; i<n; i++) {
      g[i] = CMath::max(CMath::abs(x[i] - l[i]), bound);
      z[i] = CMath::max(CMath::abs(x[i]), bound);
      t[i] = CMath::max(CMath::abs(u[i] - x[i]), bound);
      s[i] = CMath::max(CMath::abs(x[i]), bound);
    }
  }

  for (i=0, mu=0; i<n; i++)
    mu += z[i] * g[i] + s[i] * t[i];
  mu = mu / (2*n);

  /* the main loop */
  if (verb >= STATUS) {
	  SG_SDEBUG("counter | pri_inf  | dual_inf  | pri_obj   | dual_obj  | ")
	  SG_SDEBUG("sigfig | alpha  | nu \n")
	  SG_SDEBUG("-------------------------------------------------------")
	  SG_SDEBUG("---------------------------\n")
  }

  while (status == STILL_RUNNING) {
    /* predictor */

    /* put back original diagonal values */
    for (i=0; i<n; i++)
      h_x[(n+1) * i] = diag_h_x[i];

    matrix_vector(n, h_x, x, h_dot_x); /* compute h_dot_x = h_x * x */

    for (i=0; i<m; i++) {
      rho[i] = b[i];
      for (j=0; j<n; j++)
	rho[i] -= a[n*i + j] * x[j];
    }

    for (i=0; i<n; i++) {
      nu[i] = l[i] - x[i] + g[i];
      tau[i] = u[i] - x[i] - t[i];

      sigma[i] = c[i] - z[i] + s[i] + h_dot_x[i];
      for (j=0; j<m; j++)
	sigma[i] -= a[n*j + i] * y[j];

      gamma_z[i] = - z[i];
      gamma_s[i] = - s[i];
    }

    /* instrumentation */
    x_h_x = 0;
    primal_inf = 0;
    dual_inf = 0;

    for (i=0; i<n; i++) {
      x_h_x += h_dot_x[i] * x[i];
      primal_inf += CMath::sq(tau[i]);
      primal_inf += CMath::sq(nu[i]);
      dual_inf += CMath::sq(sigma[i]);
    }
    for (i=0; i<m; i++)
      primal_inf += CMath::sq(rho[i]);
    primal_inf = sqrt(primal_inf)/b_plus_1;
    dual_inf = sqrt(dual_inf)/c_plus_1;

    primal_obj = 0.5 * x_h_x;
    dual_obj = -0.5 * x_h_x;
    for (i=0; i<n; i++) {
      primal_obj += c[i] * x[i];
      dual_obj += l[i] * z[i] - u[i] * s[i];
    }
    for (i=0; i<m; i++)
      dual_obj += b[i] * y[i];

    sigfig = log10(CMath::abs(primal_obj) + 1) -
             log10(CMath::abs(primal_obj - dual_obj));
    sigfig = CMath::max(sigfig, 0.0);

    /* the diagnostics - after we computed our results we will
       analyze them */

    if (counter > counter_max) status = ITERATION_LIMIT;
    if (sigfig  > sigfig_max)  status = OPTIMAL_SOLUTION;
    if (primal_inf > 10e100)   status = PRIMAL_INFEASIBLE;
    if (dual_inf > 10e100)     status = DUAL_INFEASIBLE;
    if ((primal_inf > 10e100) & (dual_inf > 10e100)) status = PRIMAL_AND_DUAL_INFEASIBLE;
    if (CMath::abs(primal_obj) > 10e100) status = PRIMAL_UNBOUNDED;
    if (CMath::abs(dual_obj) > 10e100) status = DUAL_UNBOUNDED;

    /* write some nice routine to enforce the time limit if you
       _really_ want, however it's quite useless as you can compute
       the time from the maximum number of iterations as every
       iteration costs one cholesky decomposition plus a couple of
       backsubstitutions */

    /* generate report */
    if ((verb >= FLOOD) | ((verb == STATUS) & (status != 0)))
     SG_SDEBUG("%7i | %.2e | %.2e | % .2e | % .2e | %6.3f | %.4f | %.2e\n",
	     counter, primal_inf, dual_inf, primal_obj, dual_obj,
	     sigfig, alfa, mu);

    counter++;

    if (status == 0) {		/* we may keep on going, otherwise
				   it'll cost one loop extra plus a
				   messed up main diagonal of h_x */
      /* intermediate variables (the ones with hat) */
      for (i=0; i<n; i++) {
	hat_nu[i] = nu[i] + g[i] * gamma_z[i] / z[i];
	hat_tau[i] = tau[i] - t[i] * gamma_s[i] / s[i];
	/* diagonal terms */
	d[i] = z[i] / g[i] + s[i] / t[i];
      }

      /* initialization before the cholesky solver */
      for (i=0; i<n; i++) {
	h_x[(n+1)*i] = diag_h_x[i] + d[i];
	c_x[i] = sigma[i] - z[i] * hat_nu[i] / g[i] -
	  s[i] * hat_tau[i] / t[i];
      }
      for (i=0; i<m; i++) {
	c_y[i] = rho[i];
	for (j=i; j<m; j++)
	  h_y[m*i + j] = 0;
      }

      /* and do it */
      if (!solve_reduced(n, m, h_x, h_y, a, delta_x, delta_y, c_x, c_y, workspace, PREDICTOR))
	  {
		  status=INCONSISTENT;
		  goto exit_optimizer;
	  }

      for (i=0; i<n; i++) {
	/* backsubstitution */
	delta_s[i] = s[i] * (delta_x[i] - hat_tau[i]) / t[i];
	delta_z[i] = z[i] * (hat_nu[i] - delta_x[i]) / g[i];

	delta_g[i] = g[i] * (gamma_z[i] - delta_z[i]) / z[i];
	delta_t[i] = t[i] * (gamma_s[i] - delta_s[i]) / s[i];

	/* central path (corrector) */
	gamma_z[i] = mu / g[i] - z[i] - delta_z[i] * delta_g[i] / g[i];
	gamma_s[i] = mu / t[i] - s[i] - delta_s[i] * delta_t[i] / t[i];

	/* (some more intermediate variables) the hat variables */
	hat_nu[i] = nu[i] + g[i] * gamma_z[i] / z[i];
	hat_tau[i] = tau[i] - t[i] * gamma_s[i] / s[i];

	/* initialization before the cholesky */
	c_x[i] = sigma[i] - z[i] * hat_nu[i] / g[i] - s[i] * hat_tau[i] / t[i];
      }

      for (i=0; i<m; i++) {	/* comput c_y and rho */
	c_y[i] = rho[i];
	for (j=i; j<m; j++)
	  h_y[m*i + j] = 0;
      }

      /* and do it */
      solve_reduced(n, m, h_x, h_y, a, delta_x, delta_y, c_x, c_y, workspace,
		    CORRECTOR);

      for (i=0; i<n; i++) {
	/* backsubstitution */
	delta_s[i] = s[i] * (delta_x[i] - hat_tau[i]) / t[i];
	delta_z[i] = z[i] * (hat_nu[i] - delta_x[i]) / g[i];

	delta_g[i] = g[i] * (gamma_z[i] - delta_z[i]) / z[i];
	delta_t[i] = t[i] * (gamma_s[i] - delta_s[i]) / s[i];
      }

      alfa = -1;
      for (i=0; i<n; i++) {
	alfa = CMath::min(alfa, delta_g[i]/g[i]);
	alfa = CMath::min(alfa, delta_t[i]/t[i]);
	alfa = CMath::min(alfa, delta_s[i]/s[i]);
	alfa = CMath::min(alfa, delta_z[i]/z[i]);
      }
      alfa = (margin - 1) / alfa;

      /* compute mu */
      for (i=0, mu=0; i<n; i++)
	mu += z[i] * g[i] + s[i] * t[i];
      mu = mu / (2*n);
      mu = mu * CMath::sq((alfa - 1) / (alfa + 10));

      for (i=0; i<n; i++) {
	x[i] += alfa * delta_x[i];
	g[i] += alfa * delta_g[i];
	t[i] += alfa * delta_t[i];
	z[i] += alfa * delta_z[i];
	s[i] += alfa * delta_s[i];
      }

      for (i=0; i<m; i++)
	y[i] += alfa * delta_y[i];
    }
  }

exit_optimizer:
  if ((status == 1) && (verb >= STATUS)) {
	  SG_SDEBUG("----------------------------------------------------------------------------------\n")
	  SG_SDEBUG("optimization converged\n")
  }

  /* free memory */
  SG_FREE(workspace);
  SG_FREE(diag_h_x);
  SG_FREE(h_y);
  SG_FREE(c_x);
  SG_FREE(c_y);
  SG_FREE(h_dot_x);

  SG_FREE(rho);
  SG_FREE(nu);
  SG_FREE(tau);
  SG_FREE(sigma);
  SG_FREE(gamma_z);
  SG_FREE(gamma_s);

  SG_FREE(hat_nu);
  SG_FREE(hat_tau);

  SG_FREE(delta_x);
  SG_FREE(delta_y);
  SG_FREE(delta_s);
  SG_FREE(delta_z);
  SG_FREE(delta_g);
  SG_FREE(delta_t);

  SG_FREE(d);

  /* and return to sender */
  return status;
}
}
