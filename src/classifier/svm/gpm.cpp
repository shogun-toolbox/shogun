/******************************************************************************
 ***        GPDT - Gradient Projection Decomposition Technique              ***
 ******************************************************************************
 ***                                                                        ***
 *** GPDT is a C++ software designed to train large-scale Support Vector    ***
 *** Machines for binary classification in both scalar and distributed      ***
 *** memory parallel environments. It uses the Joachims' problem            ***
 *** decomposition technique to split the whole quadratic programming (QP)  ***
 *** problem into a sequence of smaller QP subproblems, each one being      ***
 *** solved by a suitable gradient projection method (GPM). The currently   ***
 *** implemented GPMs are the Generalized Variable Projection Method (GVPM, ***
 *** by T. Serafini, G. Zanghirati, L. Zanni) and the Dai-Fletcher method   ***
 *** (DFGPM, by Y.H. Dai, R. Fletcher).                                     ***
 ***                                                                        ***
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
 *** Copyright (C) 2004 by T. Serafini, G. Zanghirati, L. Zanni.            ***
 ***                                                                        ***
 ***                     COPYRIGHT NOTIFICATION                             ***
 ***                                                                        ***
 *** Permission to copy and modify this software and its documentation      ***
 *** for internal research use is granted, provided that this notice is     ***
 *** retained thereon and on all copies or modifications. The authors and   ***
 *** their respective Universities makes no representations as to the       ***
 *** suitability and operability of this software for any purpose. It is    ***
 *** provided "as is" without express or implied warranty.                  ***
 *** Use of this software for commercial purposes is expressly prohibited   ***
 *** without contacting the authors.                                        ***
 ***                                                                        ***
 *** This program is free software; you can redistribute it and/or modify   ***
 *** it under the terms of the GNU General Public License as published by   ***
 *** the Free Software Foundation; either version 2 of the License, or      ***
 *** (at your option) any later version.                                    ***
 ***                                                                        ***
 *** This program is distributed in the hope that it will be useful,        ***
 *** but WITHOUT ANY WARRANTY; without even the implied warranty of         ***
 *** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          ***
 *** GNU General Public License for more details.                           ***
 ***                                                                        ***
 *** You should have received a copy of the GNU General Public License      ***
 *** along with this program; if not, write to the Free Software            ***
 *** Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.              ***
 ***                                                                        ***
 *** File:     gpm.cpp                                                      ***
 *** Type:     scalar                                                       ***
 *** Version:  0.9 beta                                                     ***
 *** Date:     July 21, 2004                                                ***
 *** Revision: 1                                                            ***
 ***                                                                        ***
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gpdt.h"

extern unsigned int Randnext;

#define maxvpm    15000  /* max number of method iterations allowed  */
#define in        8000   /* max size of the QP problem to solve      */
#define alpha_max 1e10
#define alpha_min 1e-10
#define ThRand    (Randnext = Randnext * 1103515245L + 12345L)
#define ThRandPos ((Randnext = Randnext * 1103515245L + 12345L) & 0x7fffffff)

/* Uncomment if you want to allocate vectors on the stack  *
 * instead of the heap. On some architectures this helps   *
 * improving speed, but may generate a stack overflow      */
// #define VARIABLES_ON_STACK

/* Uncomment if you want to use the adaptive steplength    *
   in the GVPM solver                                      */
#define VPM_ADA

/* Uncomment if you want to use the scaled versions of     *
 * the GVPM and Dai-Fletcher solvers                                */
// #define VS_SCALED

/*** utility routines prototyping ***/
//double max(double x,double y);
int    Pardalos   (int n, int *iy, double e, double *qk, double *dia,
                                   double l, double u,   double *x);
void   quick_sort (double a[], int n);
double dnrm2      (int n, double t[]);

/******************************************************************************/
/*** Encapsulating method to call the chosen Gradient Projection Method     ***/
/******************************************************************************/
int gpm_solver(int Solver, int n, float *A, double *b, double c, double e,
               int *iy, double *x, double tol, int *ls)
{
  /*** Uncomment the following if you need to scale the QP Hessian matrix
       before calling the chosen solver
  int    i, j;
  float  *ptrA;
  double max, s;

  max = fabs(A[0][0]);
  for (i = 1; i < n; i++)
      if (fabs(A[i][i]) > max)
          max = fabs(A[i][i]);

  s    = 1.0 / max;
  ptrA = vecA;
  for (i = 0; i < n; i++)
      for (j = 0;j < n; j++)
          *ptrA++ = (float)(A[i][j]*s);

  if (Solver == SOLVER_FLETCHER)
      j = FletcherAlg2A(n, vecA, b, c/s, e/s, iy, x, tol, ls);
  else
      j = vpmkt(n, vecA, b, c/s, e/s, iy, x, tol, ls);

  for (i = 0; i < n; i++)
      x[i] *= s;
  ***/

  /*** calling the chosen solver with unscaled data ***/
  if (Solver == SOLVER_FLETCHER)
    return FletcherAlg2A(n, A, b, c, e, iy, x, tol, ls);
  else
    return vpmkt(n, A, b, c, e, iy, x, tol, ls);
}

/******************************************************************************/
/*** Variable Projection Method (VPM, by L. Zanni and V. Ruggiero, 2000)    ***/
/******************************************************************************/
int vpmkt(int n, float *vecA, double *b, double c, double e, int *iy,
          double *x, double tol, int *ls)
{
  int     i, j, iter, it, luv, info;
  double  gd, max, normd, dAd, lam, alpha, kktlam, ak, bk;

  int     lscount = 0;
  double  eps     = 1.0e-16;
  double  DELTAsv = EPS_SV * c;

  /* solver-specific settings */
#ifdef VPM_ADA
  int    nc = 1, ia1 = -1;
  double alpha1, alpha2;
#endif

  /* allocation-dependant settings */
#ifdef VARIABLES_ON_STACK

  int    ipt[in], ipt2[in], uv[in];
  double g[in], y[in], tempv[in], d[in], Ad[in], diaga[in], t[in];

#else

  int    *ipt, *ipt2, *uv;
  double *g, *y, *tempv, *d, *Ad, *diaga, *t;

  /*** array allocations ***/
  ipt   = (int    *)malloc(n * sizeof(int   ));
  ipt2  = (int    *)malloc(n * sizeof(int   ));
  uv    = (int    *)malloc(n * sizeof(int   ));
  g     = (double *)malloc(n * sizeof(double));
  y     = (double *)malloc(n * sizeof(double));
  tempv = (double *)malloc(n * sizeof(double));
  d     = (double *)malloc(n * sizeof(double));
  Ad    = (double *)malloc(n * sizeof(double));
  diaga = (double *)malloc(n * sizeof(double));
  t     = (double *)malloc(n * sizeof(double));

#endif

  Randnext = 1234;

#ifdef VS_SCALED

  for (i = 0; i < n; i++)
      diaga[i] = (double)vecA[i*n+i];

#else

  for (i = 0; i < n; i++)
      diaga[i] = 1.0;

#endif

//  memset(x, 0, n*sizeof(double));
//  memset(tempv, 0, n*sizeof(double));

  for (i = 0; i < n; i++)
      tempv[i] = -x[i];

  Pardalos(n, iy, e, tempv, diaga, 0, c, x);

  /* compute g = A*x + b in sparse form          *
   * (inline computation for better perfomrance) */
  {
    int   it;
    float *tempA;

    it = 0;
    for (i = 0; i < n; i++)
        if (fabs(x[i]) > DELTAsv*1e-2)
            ipt[it++] = i;

    memset(t, 0, n*sizeof(double));
    for (i = 0; i < it; i++)
    {
        tempA = vecA + ipt[i]*n;
        for (j = 0; j < n; j++)
            t[j] += (tempA[j]*x[ipt[i]]);
    }
  }

  for (i = 0; i < n; i++)
  {
    g[i] = t[i] + b[i],
    y[i] = g[i] - diaga[i]*x[i];
  }

  Pardalos(n, iy, e, y, diaga, 0, c, tempv);

  max = alpha_min;
  for (i = 0; i < n; i++)
  {
      y[i] = sqrt(diaga[i]) * (tempv[i] - x[i]);
      if (fabs(y[i]) > max)
          max = fabs(y[i]);
  }

  if (max < c*tol*1e-3)
  {
      lscount = 0;
      iter    = 0;
      goto Clean;
  }

  alpha = 1.0 / max;

  for (iter = 1; iter <= maxvpm; iter++)
  {
      for (i = 0; i < n; i++)
          tempv[i] = alpha*g[i] - diaga[i]*x[i];

      it = Pardalos(n, iy, e, tempv, diaga, 0, c, y);

      gd = 0.0;
      for (i = 0; i < n; i++)
      {
          d[i] = y[i] - x[i];
          gd  += d[i] * g[i];
      }

      /* compute Ad = A*d  or  Ad = Ay-t depending on their sparsity  *
       * (inline computation for better perfomrance)                  */
      {
         int   i, it, it2;
         float *tempA;

         it = it2 = 0;
         for (i = 0; i < n; i++)
             if (fabs(d[i]) > (DELTAsv*1.0e-2))
                 ipt[it++] = i;
         for (i = 0; i < n; i++)
             if (fabs(y[i]) > DELTAsv)
                 ipt2[it2++] = i;

         memset(Ad, 0, n*sizeof(double));
         if (it < it2) // Ad = A*d
         {
            for (i = 0; i < it; i++)
            {
                tempA = vecA + ipt[i]*n;
                for (j = 0; j < n; j++)
                    Ad[j] += (tempA[j]*d[ipt[i]]);
            }
         }
         else          // Ad = A*y-t
         {
            for (i = 0; i < it2; i++)
            {
                tempA = vecA + ipt2[i]*n;
                for (j = 0; j < n; j++)
                    Ad[j] += (tempA[j]*y[ipt2[i]]);
            }
            for (j = 0; j < n; j++)
                Ad[j] -= t[j];
         }
      }

      normd = 0.0;
      for (i = 0; i < n; i++)
          normd += d[i] * d[i] * diaga[i];

      dAd = 0.0;
      for (i = 0; i < n; i++)
          dAd += d[i]*Ad[i];

      if (dAd > eps*normd && gd < 0.0)
      {
          lam = -gd/dAd;
          if (lam > 1.0 || lam < 0.0)
              lam = 1.0;
          else
              lscount++;

#ifdef VPM_ADA

          /*** use the adaptive switching rule for steplength selection ***/

          // compute alpha1 = (d'* (d.*diaga)) / (d'*Ad);
          alpha1 = normd / dAd;

          // alpha2 = d'*Ad / (Ad' * (Ad./diaga));
          alpha2 = 0.0;
          for (i = 0; i < n; i++)
               alpha2 += Ad[i] * Ad[i] / diaga[i];
          alpha2 = dAd / alpha2;

          if ( (nc > 2
                && (
                     (ia1 == 1
                      && (
                           lam < 0.1 || (alpha1 > alpha && alpha2 < alpha)
                         )
                     )
                     ||
                     (ia1 == -1
                      && (
                           lam > 5.0 || (alpha1 > alpha && alpha2 < alpha)
                         )
                     )
                   )
               )
               || nc > 9 )
          {
              ia1 = -ia1;
              nc  = 0;
          }

          if (ia1 == 1)
              alpha = alpha1;
          else
              alpha = alpha2;

          if (alpha < alpha_min)
              alpha = alpha_min;
          else if (alpha > alpha_max)
              alpha = alpha_max;

          nc++;

#else

          /*** use the fixed switching rule for steplength selection ***/

          if ((iter % 6) < 3) // alpha = d'*Ad / (Ad' * (Ad./diaga));
          {
              alpha = 0.0;
              for (i = 0; i < n; i++)
                  alpha += Ad[i] * Ad[i] / diaga[i];
              alpha = dAd / alpha;
          }
          else                // alpha = (d'* (d.*diaga)) / (d'*Ad);
          {
              alpha = 0.0;
              for (i = 0; i < n; i++)
                  alpha += d[i] * d[i] * diaga[i];
              alpha = alpha / dAd;
          }

#endif

      }
      else
      {
          lam   = 1.0;
          alpha = alpha_max;
      }

      for (i = 0; i < n; i++)
      {
          x[i] = x[i] + lam*d[i];
          t[i] = t[i] + lam*Ad[i];
          g[i] = b[i] + t[i];
      }

      /*** stopping criterion based on KKT conditions ***/
      bk = 0.0;
      ak = 0.0;
      for (i = 0; i < n; i++)
      {
          bk +=  x[i] * x[i];
          ak +=  d[i] * d[i];
      }

      if (lam*sqrt(ak) < tol*10 * sqrt(bk))
      {
          it     = 0;
          luv    = 0;
          kktlam = 0.0;
          for (i = 0; i < n; i++)
          {
              if (x[i] > DELTAsv && x[i] < c-DELTAsv)
              {
                  ipt[it++] = i;
                  kktlam    = kktlam - iy[i]*g[i];
              }
              else
                  uv[luv++] = i;
          }

          if (it == 0)
          {
              if (lam*sqrt(ak) < tol*0.5 * sqrt(bk))
              goto Clean;
          }
          else
          {
              kktlam = kktlam/it;
              info   = 1;
              for (i = 0; i < it; i++)
                  if (fabs(iy[ipt[i]]*g[ipt[i]]+kktlam) > tol)
                  {
                      info = 0;
                      break;
                  }

              if (info == 1)
                  for (i = 0; i < luv; i++)
                      if (x[uv[i]] <= DELTAsv)
                      {
                          if (g[uv[i]] + kktlam*iy[uv[i]] < -tol)
                          {
                              info = 0;
                              break;
                          }
                      }
                      else
                      {
                          if (g[uv[i]] + kktlam*iy[uv[i]] > tol)
                          {
                              info = 0;
                              break;
                          }
                      }

              if (info == 1)
                  goto Clean;
          }
      } // stopping rule based on the norm of d_k
  }

  printf("VPM exits after maxvpm = %d iterations.\n", maxvpm);

Clean:

  /*** allocation-dependant freeing ***/
#ifndef VARIABLES_ON_STACK
  free(t);
  free(uv);
  free(ipt2);
  free(ipt);
  free(g);
  free(y);
  free(tempv);
  free(d);
  free(diaga);
  free(Ad);
#endif

  if (ls != NULL)
      *ls = lscount;
  return(iter);
}

/******************************************************************************/
/*** Dai-Fletcher QP solver (Y. Dai, R. Fletcher, 2003)                     ***/
/******************************************************************************/
int FletcherAlg2A(int n, float *vecA, double *b, double c, double e,
                  int *iy, double *x, double tol, int *ls)
{
  int    i, j, iter, it, luv, info, lscount = 0;
  double gd, max, ak, bk, akold, bkold, lamnew, alpha, kktlam;
  double eps     = 1.0e-16;
  double DELTAsv = EPS_SV * c;

  /*** variables for the adaptive nonmonotone linesearch ***/
  int    L, llast;
  double fr, fbest, fv, fc, fv0;

  /*** allocation-dependant settings ***/
#ifdef VARIABLES_ON_STACK

  int    ipt[in], ipt2[in], uv[in];
  double g[in], y[in], tempv[in], d[in], Ad[in], diaga[in], t[in],
         xplus[in], tplus[in], sk[in], yk[in];
#else

  int    *ipt, *ipt2, *uv;
  double *g, *y, *tempv, *d, *Ad, *diaga, *t, *xplus, *tplus, *sk, *yk;

  /*** arrays allocation ***/
  ipt   = (int    *)malloc(n * sizeof(int   ));
  ipt2  = (int    *)malloc(n * sizeof(int   ));
  uv    = (int    *)malloc(n * sizeof(int   ));
  g     = (double *)malloc(n * sizeof(double));
  y     = (double *)malloc(n * sizeof(double));
  tempv = (double *)malloc(n * sizeof(double));
  d     = (double *)malloc(n * sizeof(double));
  Ad    = (double *)malloc(n * sizeof(double));
  diaga = (double *)malloc(n * sizeof(double));
  t     = (double *)malloc(n * sizeof(double));
  xplus = (double *)malloc(n * sizeof(double));
  tplus = (double *)malloc(n * sizeof(double));
  sk    = (double *)malloc(n * sizeof(double));
  yk    = (double *)malloc(n * sizeof(double));

#endif

  Randnext = 1234;

#ifdef VS_SCALED
  for (i = 0; i < n; i++)
      diaga[i] = (double)vecA[i*n+i];
#else
  for (i = 0; i < n; i++)
      diaga[i] = 1.0;
#endif

  //memset(tempv, 0, n*sizeof(double));
  for (i = 0; i < n; i++)
      tempv[i] = -x[i];

  /*** call the Pardalos-Kovoor solver ***/
  Pardalos(n, iy, e, tempv, diaga, 0, c, x);

  // g = A*x + b;
  // SparseProd(n, t, A, x, ipt);
  {
    int   it;
    float *tempA;

    it = 0;
    for (i = 0; i < n; i++)
        if (fabs(x[i]) > DELTAsv)
            ipt[it++] = i;

    memset(t, 0, n*sizeof(double));
    for (i = 0; i < it; i++)
    {
         tempA = vecA + ipt[i] * n;
         for (j = 0; j < n; j++)
             t[j] += (tempA[j]*x[ipt[i]]);
    }
  }

  for (i = 0; i < n; i++)
  {
    g[i] = t[i] + b[i],
    y[i] = g[i] - diaga[i]*x[i];
  }

  /*** call the Pardalos-Kovoor solver ***/
  Pardalos(n, iy, e, y, diaga, 0, c, tempv);

  max = alpha_min;
  for (i = 0; i < n; i++)
  {
      y[i] = sqrt(diaga[i]) * (tempv[i] - x[i]);
      if (fabs(y[i]) > max)
          max = fabs(y[i]);
  }

  if (max < c*tol*1e-3)
  {
      lscount = 0;
      iter    = 0;
      goto Clean;
  }

  alpha = 1.0 / max;

  fv0   = 0.0;
  for (i = 0; i < n; i++)
      fv0 += x[i] * (0.5*t[i] + b[i]);

  /*** adaptive nonmonotone linesearch ***/
  L     = 2;
  fr    = alpha_max;
  fbest = fv0;
  fc    = fv0;
  llast = 0;
  akold = bkold = 0.0;

  for (iter = 1; iter <= maxvpm; iter++)
  {
      for (i = 0; i < n; i++)
          tempv[i] = alpha*g[i] - diaga[i]*x[i];

      it = Pardalos(n, iy, e, tempv, diaga, 0, c, y);

      gd = 0.0;
      for (i = 0; i < n; i++)
      {
          d[i] = y[i] - x[i];
          gd  += d[i] * g[i];
      }

      /* compute Ad = A*d  or  Ad = Ay-t depending on their sparsity */
      {
         int   i, it, it2;
         float *tempA;

         it = it2 = 0;
         for (i = 0; i < n; i++)
             if (fabs(d[i]) > (DELTAsv*1.0e-2))
                 ipt[it++]   = i;
         for (i = 0; i < n; i++)
             if (fabs(y[i]) > DELTAsv)
                 ipt2[it2++] = i;

         memset(Ad, 0, n*sizeof(double));
         if (it < it2) // compute Ad = A*d
         {
            for (i = 0; i < it; i++)
            {
                tempA = vecA + ipt[i]*n;
                for (j = 0; j < n; j++)
                    Ad[j] += (tempA[j]*d[ipt[i]]);
            }
         }
         else          // compute Ad = A*y-t
         {
            for (i = 0; i < it2; i++)
            {
                tempA = vecA + ipt2[i]*n;
                for (j = 0; j < n; j++)
                    Ad[j] += (tempA[j]*y[ipt2[i]]);
            }
            for (j = 0; j < n; j++)
                Ad[j] -= t[j];
         }
      }

      ak = 0.0;
      for (i = 0; i < n; i++)
          ak += d[i] * d[i] * diaga[i];

      bk = 0.0;
      for (i = 0; i < n; i++)
          bk += d[i]*Ad[i];

      if (bk > eps*ak && gd < 0.0)    // ak is normd
          lamnew = -gd/bk;
      else
          lamnew = 1.0;

      fv = 0.0;
      for (i = 0; i < n; i++)
      {
          xplus[i] = x[i] + d[i];
          tplus[i] = t[i] + Ad[i];
          fv      += xplus[i] * (0.5*tplus[i] + b[i]);
      }

      if ((iter == 1 && fv >= fv0) || (iter > 1 && fv >= fr))
      {
          lscount++;
          if (lamnew > 1 || lamnew < 0)
              printf("\n Warning: lamnew = %lf\n", lamnew);
          fv = 0.0;
          for (i = 0; i < n; i++)
          {
              xplus[i] = x[i] + lamnew*d[i];
              tplus[i] = t[i] + lamnew*Ad[i];
              fv      += xplus[i] * (0.5*tplus[i] + b[i]);
          }
      }

      for (i = 0; i < n; i++)
      {
          sk[i] = xplus[i] - x[i];
          yk[i] = tplus[i] - t[i];
          x[i]  = xplus[i];
          t[i]  = tplus[i];
          g[i]  = t[i] + b[i];
      }

      // update the line search control parameters

      if (fv < fbest)
      {
          fbest = fv;
          fc    = fv;
          llast = 0;
      }
      else
      {
          fc = (fc > fv ? fc : fv);
          llast++;
          if (llast == L)
          {
              fr    = fc;
              fc    = fv;
              llast = 0;
          }
      }

      ak = bk = 0.0;
      for (i = 0; i < n; i++)
      {
          ak += sk[i]*sk[i];
          bk += sk[i]*yk[i];
      }

      if (bk < eps*ak)
          alpha = alpha_max;
      else
      {
          if (bkold < eps*akold)
              alpha = ak/bk;
          else
              alpha = (akold+ak)/(bkold+bk);

          if (alpha > alpha_max)
              alpha = alpha_max;
          else if (alpha < alpha_min)
              alpha = alpha_min;
      }
      akold = ak;
      bkold = bk;

      /*** stopping criterion based on KKT conditions ***/

      bk = 0.0;
      for (i = 0; i < n; i++)
          bk +=  x[i] * x[i];

      if (sqrt(ak) < tol*10 * sqrt(bk))
      {
          it     = 0;
          luv    = 0;
          kktlam = 0.0;
          for (i = 0; i < n; i++)
          {
              if (x[i]>DELTAsv && x[i]<c-DELTAsv)
              {
                  ipt[it++] = i;
                  kktlam    = kktlam - iy[i]*g[i];
              }
              else
                  uv[luv++] = i;
          }

          if (it == 0)
          {
              if (sqrt(ak) < tol*0.5 * sqrt(bk))
                  goto Clean;
          }
          else
          {

              kktlam = kktlam/it;
              //kktlam = -iy[ipt[0]]*g[ipt[0]];
              info   = 1;
              for (i=0; i<it; i++)
                  if (fabs(iy[ipt[i]]*g[ipt[i]]+kktlam) > tol)
                  {
                      info = 0;
                      break;
                  }

              if (info == 1)
                  for (i = 0; i < luv; i++)
                      if (x[uv[i]] <= DELTAsv)
                      {
                          if (g[uv[i]] + kktlam*iy[uv[i]] < -tol)
                          {
                              info = 0;
                              break;
                          }
                      }
                      else
                      {
                          if (g[uv[i]] + kktlam*iy[uv[i]] > tol)
                          {
                              info = 0;
                              break;
                          }
                      }

              if (info == 1)
                  goto Clean;
          }
      }
  }

  printf("\nDai-Fletcher method exits after maxvpm = %d iterations.\n", maxvpm);

Clean:

#ifndef VARIABLES_ON_STACK
  free(sk);
  free(yk);
  free(tplus);
  free(xplus);
  free(t);
  free(uv);
  free(ipt2);
  free(ipt);
  free(g);
  free(y);
  free(tempv);
  free(d);
  free(diaga);
  free(Ad);
#endif

  if (ls != NULL)
      *ls = lscount;
  return(iter);

}

/******************************************************************************/
/*** Pardalos-Kovoor method                                                 ***/
/******************************************************************************/
int Pardalos(int n, int *iy, double e,   double *qk, double *dia,
                             double low, double up,  double *x)
{
  /***********************************************************************
   * This method solves the problem                                      *
   *                    min     x'*diag(dia)*x/2 + qk'*x                 *
   *                    subj to iy'*x = e                                *
   *                            l <= x <= u                              *
   *                            iy(i) ~= 0                               *
   * by the Pardalos-Kovoor algorithm [Math. Prog. 46 (1990), 321-328].  *
   ***********************************************************************/

  int    i, l, iter; /* conters    */
  int    luv, lxint; /* dimensions */
  double d, xmin, xmax, xmold, xmid, xx, ts, sw, s, s1, testsum;

  /*** allocation-dependant settings ***/
#ifdef VARIABLES_ON_STACK
  int    uv[in], uvt[in];
  double xint[2*in+2], xint2[2*in+2], a[in], b[in], at[in], bt[in];
  double newdia[in], newdt[in];
#else

  int    *uv, *uvt;
  double *xint, *xint2, *a, *b, *at, *bt, *newdia, *newdt;

  /*** arrays allocation ***/
  uv     = (int    *)malloc(n * sizeof(int           ));
  uvt    = (int    *)malloc(n * sizeof(int           ));
  a      = (double *)malloc(n * sizeof(double        ));
  b      = (double *)malloc(n * sizeof(double        ));
  at     = (double *)malloc(n * sizeof(double        ));
  bt     = (double *)malloc(n * sizeof(double        ));
  newdia = (double *)malloc(n * sizeof(double        ));
  newdt  = (double *)malloc(n * sizeof(double        ));
  xint   = (double *)malloc((2*n + 2) * sizeof(double));
  xint2  = (double *)malloc((2*n + 2) * sizeof(double));

#endif

  d = 0.0;
  for (i = 0; i < n; i++)
      d += iy[i] * qk[i] / dia[i];
  d = 0.5 * (d-e);

  for (i = 0; i < n; i++)
  {
      /* The following computations should divide by iy[i] instead           *
       * of multiply by iy[i], but this is correct for binary classification *
       * with labels -1 and 1.                                               */
      if (iy[i] > 0)
      {
          a[i] = ((qk[i] + dia[i]*low) * iy[i]) * 0.5;
          b[i] = ((dia[i]*up + qk[i]) * iy[i]) * 0.5;
      }
      else
      {
          b[i] = ((qk[i] + dia[i]*low) * iy[i]) * 0.5;
          a[i] = ((dia[i]*up + qk[i]) * iy[i]) * 0.5;
      }
      newdia[i] = (iy[i]*iy[i]) / dia[i];
  }

  xmin = -1e33;
  xmax = 1e33;

  /* arrays initialization */
  for (i = 0; i < n; i++)
  {
      uv[i]     = i;    /* contains the unset variables */
      xint[i]   = a[i];
      xint[n+i] = b[i];
  }

  xmid        = xmin;
  xint[2*n]   = xmin;
  xint[2*n+1] = xmax;
  ts          = 0.0;
  sw          = 0.0;
  luv         = n;
  lxint       = 2*n+2;

  /* sort the variables in ascending order */
  quick_sort(xint,lxint);

  /**********************************************************************/
  /* Loop implementing the Joachim's method to choose the new variables */
  /* entering the working set. It ends when luv = 0.                    */
  iter = 0;
  do {
     for (i = 0; i < luv; i++)
     {
         at[i]    = a[uv[i]];
         bt[i]    = b[uv[i]];
         newdt[i] = newdia[uv[i]];
     }
     xmold = xmid;

     if ((lxint % 2) == 0)
         xmid = (xint[lxint/2-1] + xint[lxint/2])/2;
     else
         xmid = xint[lxint/2];

     if (xmold == xmid)
         xmid = xint[(int)(ThRandPos % lxint)];

     s  = ts;
     s1 = sw;
     for (i = 0; i < luv; i++)
     {
         if (xmid > bt[i])
             s  += newdt[i]*bt[i];
         else if (xmid < at[i])
             s  += newdt[i]*at[i];
         else
             s1 += newdt[i];
     }

     testsum = s + s1*xmid;
     if (testsum <= (d+(1e-15)))
         xmin = xmid;
     if (testsum >= (d-(1e-15)))
         xmax = xmid;

     l = 0;
     for (i = 0; i < lxint; i++)
         if((xint[i] >= xmin) && (xint[i] <= xmax))
            xint2[l++] = xint[i];
     lxint = l;
     memcpy(xint, xint2, lxint*sizeof(double));

     l = 0;
     for (i = 0; i < luv; i++)
     {
         if (xmin >= bt[i])
             ts += newdt[i]*bt[i];
         else if (xmax <= at[i])
             ts += newdt[i]*at[i];
         else if ((at[i] <= xmin) && (bt[i] >= xmax))
             sw += newdt[i];
         else
             uvt[l++] = uv[i];
    }
    luv = l;
    memcpy(uv, uvt, luv*sizeof(int));
    iter++;
  } while(luv != 0);
  /**********************************************************************/

  if (sw == 0)
      xx = xmin;
  else
      xx = (d-ts) / sw;

  for (i = 0; i < n; i++)
  {
      if (b[i] <= xmin)
          x[i] = b[i];
      else if (a[i] >= xmax)
          x[i] = a[i];
      else if ((a[i]<=xmin) && (xmax<=b[i]))
          x[i] = xx;
      else
          printf("\nWarning: inner solver troubles...\n");
  }

  for (i = 0; i < n; i++)
      x[i] = (2.0*x[i]*iy[i]-qk[i]) / dia[i];

#ifndef VARIABLES_ON_STACK
  free(newdt);
  free(newdia);
  free(a);
  free(b);
  free(uvt);
  free(uv);
  free(bt);
  free(at);
  free(xint2);
  free(xint);
#endif

  return(iter);
}

/******************************************************************************/
/*** Quick sort function                                                    ***/
/******************************************************************************/
void quick_sort(double a[], int n)
{
  /************************************************************************
   * Here two partitions are used. At each step the smaller is considered *
   * first: this way the sizing of arrays pd[] (right partition) and      *
   * ps[] (left partition) is enough to sort up to 1000000 components.    *
   ************************************************************************/
  int    i, j, s, d, l, ps[20], pd[20];
  double x, w;

  l     = 0;
  ps[0] = 0;
  pd[0] = n-1;

  do
  {
     s = ps[l];
     d = pd[l];
     l--;
     do
     {
        i = s;
        j = d;
        x = a[(s+d)/2];
        do
        {
           while (a[i] < x) i++;
           while (a[j] > x) j--;
           if (i <= j)
           {
               w=a[i];
               a[i]=a[j];
               i++;
               a[j]=w;
               j--;
           }
        } while (i <= j);

        if (j-s > d-i)
        {
            l++;
            ps[l] = s;
            pd[l] = j;
            s     = i;
        }
        else
        {
            if (i < d)
            {
                l++;
                ps[l] = i;
                pd[l] = d;
            }
            d = j;
         };
     } while (s < d);
  } while (l >= 0);
}

/******************************************************************************/
/*** Compute the vector Euclidean norm                                      ***/
/******************************************************************************/
double dnrm2(int n, double t[])
{
  /************************************************************************
   * This implementation is not the best one and it is provided only for  *
   * user opportunity: in most systems the far better BLAS implementation *
   * is available by linking the native blas library and it should        *
   * certainly be preferred to this one.                                  *
   ************************************************************************/
  int    i;
  double s = 0.0;
  for (i = n-1; i >= 0; i--)
      s += t[i]*t[i];
  return(sqrt(s));
}

/******************************************************************************/
/*** End of gpm.cpp file                                                    ***/
/******************************************************************************/
