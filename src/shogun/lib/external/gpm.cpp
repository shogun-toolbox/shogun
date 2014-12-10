/******************************************************************************
 ***        GPDT - Gradient Projection Decomposition Technique              ***
 ******************************************************************************
 ***                                                                        ***
 *** GPDT is a C++ software designed to train large-scale Support Vector    ***
 *** Machines for binary classification in both scalar and distributed      ***
 *** memory parallel environments. It uses the Joachims' problem            ***
 *** decomposition technique to split the whole quadratic programming (QP)  ***
 *** problem into a sequence of smaller QP subproblems, each one being      ***
 *** solved by a suitable gradient projection method (GPM). The presently   ***
 *** implemented GPMs are the Generalized Variable Projection Method        ***
 *** GVPM (T. Serafini, G. Zanghirati, L. Zanni, "Gradient Projection       ***
 *** Methods for Quadratic Programs and Applications in Training Support    ***
 *** Vector Machines"; Optim. Meth. Soft. 20, 2005, 353-378) and the        ***
 *** Dai-Fletcher Method DFGPM (Y. Dai and R. Fletcher,"New Algorithms for  ***
 *** Singly Linear Constrained Quadratic Programs Subject to Lower and      ***
 *** Upper Bounds"; Math. Prog. to appear).                                 ***
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
 *** Copyright (C) 2004-2008 by T. Serafini, G. Zanghirati, L. Zanni.       ***
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
 *** the Free Software Foundation; either version 3 of the License, or      ***
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
 *** Version:  1.0                                                          ***
 *** Date:     October, 2005                                                ***
 *** Revision: 1                                                            ***
 ***                                                                        ***
 *** SHOGUN adaptions  Written (W) 2006-2008 Soeren Sonnenburg              ***
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/external/gpdt.h>
#include <shogun/io/SGIO.h>

namespace shogun
{
#define maxvpm           30000  /* max number of method iterations allowed  */
#define maxprojections   200
#define in               8000   /* max size of the QP problem to solve      */
#define alpha_max        1e10
#define alpha_min        1e-10

extern uint32_t Randnext;
#define ThRand    (Randnext = Randnext * 1103515245L + 12345L)
#define ThRandPos ((Randnext = Randnext * 1103515245L + 12345L) & 0x7fffffff)

int32_t InnerProjector(
	int32_t method, int32_t n, int32_t *iy, float64_t e, float64_t *qk, float64_t l,
	float64_t u, float64_t *x, float64_t &lambda);

/* Uncomment if you want to allocate vectors on the stack  *
 * instead of the heap. On some architectures this helps   *
 * improving speed, but may generate a stack overflow      */
// #define VARIABLES_ON_STACK

/* Uncomment if you want to use the adaptive steplength    *
   in the GVPM solver                                      */
#define VPM_ADA


/******************************************************************************
 *** Generalized Variable Projection Method (T. Serafini, G. Zanghirati,    ***
 *** L. Zanni, "Gradient Projection Methods for Quadratic Programs and      ***
 *** Applications in Training Support Vector Machines";                     ***
 *** Optim. Meth. Soft. 20, 2005, 353-378)                                  ***
 ******************************************************************************/
int32_t gvpm(
	int32_t Projector, int32_t n, float32_t *vecA, float64_t *b, float64_t c,
	float64_t e, int32_t *iy, float64_t *x, float64_t tol, int32_t *ls,
	int32_t *proj)
{
  int32_t i, j, iter, it, it2, luv, info;
  float64_t  gd, max, normd, dAd, lam, lamnew, alpha, kktlam, ak, bk;

  int32_t lscount = 0, projcount = 0;
  float64_t  eps     = 1.0e-16;
  float64_t  DELTAsv, ProdDELTAsv;
  float64_t  lam_ext;

  /* solver-specific settings */
#ifdef VPM_ADA
  int32_t     nc = 1, ia1 = -1;
  float64_t  alpha1, alpha2;
#endif

  /* allocation-dependant settings */
#ifdef VARIABLES_ON_STACK

  int32_t     ipt[in], ipt2[in], uv[in];
  float64_t  g[in], y[in], tempv[in], d[in], Ad[in], t[in];

#else

  int32_t     *ipt, *ipt2, *uv;
  float64_t  *g, *y, *tempv, *d, *Ad, *t;

  /*** array allocations ***/
  ipt   = SG_MALLOC(int32_t, n);
  ipt2  = SG_MALLOC(int32_t, n);
  uv    = SG_MALLOC(int32_t, n);
  g     = SG_MALLOC(float64_t, n);
  y     = SG_MALLOC(float64_t, n);
  d     = SG_MALLOC(float64_t, n);
  Ad    = SG_MALLOC(float64_t, n);
  t     = SG_MALLOC(float64_t, n);
  tempv = SG_MALLOC(float64_t, n);

#endif

  DELTAsv = EPS_SV * c;
  if (tol <= 1.0e-5 || n <= 20)
      ProdDELTAsv = 0.0F;
  else
      ProdDELTAsv = EPS_SV * c;

  for (i = 0; i < n; i++)
      tempv[i] = -x[i];
  lam_ext = 0.0;
  projcount += InnerProjector(Projector, n, iy, e, tempv, 0, c, x, lam_ext);

  /* compute g = A*x + b in sparse form          *
   * (inline computation for better perfomrance) */
  {
    float32_t *tempA;

    it = 0;
    for (i = 0; i < n; i++)
        if (fabs(x[i]) > ProdDELTAsv*1e-2)
            ipt[it++] = i;

    memset(t, 0, n*sizeof(float64_t));
    for (i = 0; i < it; i++)
    {
        tempA = vecA + ipt[i]*n;
        for (j = 0; j < n; j++)
            t[j] += (tempA[j] * x[ipt[i]]);
    }
  }

  for (i = 0; i < n; i++)
  {
    g[i] = t[i] + b[i],
    y[i] = g[i] - x[i];
  }

  projcount += InnerProjector(Projector, n, iy, e, y, 0, c, tempv, lam_ext);

  max = alpha_min;
  for (i = 0; i < n; i++)
  {
      y[i] = tempv[i] - x[i];
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
          tempv[i] = alpha*g[i] - x[i];

      projcount += InnerProjector(Projector, n, iy, e, tempv, 0, c, y, lam_ext);

      gd = 0.0;
      for (i = 0; i < n; i++)
      {
          d[i] = y[i] - x[i];
          gd  += d[i] * g[i];
      }

      /* compute Ad = A*d  or  Ad = Ay-t depending on their sparsity  *
       * (inline computation for better perfomrance)                  */
      {
         float32_t *tempA;

         it = it2 = 0;
         for (i = 0; i < n; i++)
             if (fabs(d[i]) > (ProdDELTAsv*1.0e-2))
                 ipt[it++] = i;
         for (i = 0; i < n; i++)
             if (fabs(y[i]) > ProdDELTAsv)
                 ipt2[it2++] = i;

         memset(Ad, 0, n*sizeof(float64_t));
         if (it < it2) // Ad = A*d
         {
             for (i = 0; i < it; i++)
             {
                 tempA = vecA + ipt[i]*n;
                 for (j = 0; j < n; j++)
                     Ad[j] += (tempA[j] * d[ipt[i]]);
             }
         }
         else          // Ad = A*y - t
         {
            for (i = 0; i < it2; i++)
            {
                tempA = vecA + ipt2[i]*n;
                for (j = 0; j < n; j++)
                    Ad[j] += (tempA[j] * y[ipt2[i]]);
            }
            for (j = 0; j < n; j++)
                Ad[j] -= t[j];
         }
      }

      normd = 0.0;
      for (i = 0; i < n; i++)
          normd += d[i] * d[i];

      dAd = 0.0;
      for (i = 0; i < n; i++)
          dAd += d[i]*Ad[i];

      if (dAd > eps*normd && gd < 0.0)
      {
          lam = lamnew = -gd/dAd;
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
               alpha2 += Ad[i] * Ad[i];
          alpha2 = dAd / alpha2;

          if ( (nc > 2
                && (
                     (ia1 == 1
                      && (
                           lamnew < 0.1 || (alpha1 > alpha && alpha2 < alpha)
                         )
                     )
                     ||
                     (ia1 == -1
                      && (
                           lamnew > 5.0 || (alpha1 > alpha && alpha2 < alpha)
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
                  alpha += Ad[i] * Ad[i];
              alpha = dAd / alpha;
          }
          else                // alpha = (d'* (d.*diaga)) / (d'*Ad);
          {
              alpha = 0.0;
              for (i = 0; i < n; i++)
                  alpha += d[i] * d[i];
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
			  {
                  if (fabs(iy[ipt[i]]*g[ipt[i]]+kktlam) > tol)
                  {
                      info = 0;
                      break;
                  }
			  }

              if (info == 1)
			  {
                  for (i = 0; i < luv; i++)
				  {
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
				  }
			  }

              if (info == 1)
                  goto Clean;
          }
      } // stopping rule based on the norm of d_k
  }

  SG_SWARNING("GVPM exits after maxvpm = %d iterations.\n", maxvpm)

Clean:

  /*** allocation-dependant freeing ***/
#ifndef VARIABLES_ON_STACK
  SG_FREE(t);
  SG_FREE(uv);
  SG_FREE(ipt2);
  SG_FREE(ipt);
  SG_FREE(g);
  SG_FREE(y);
  SG_FREE(tempv);
  SG_FREE(d);
  SG_FREE(Ad);
#endif

  if (ls != NULL)   *ls   = lscount;
  if (proj != NULL) *proj = projcount;
  return(iter);
}

/******************************************************************************
 *** Dai-Fletcher QP solver (Y. Dai and R. Fletcher,"New Algorithms for     ***
 *** Singly Linear Constrained Quadratic Programs Subject to Lower and      ***
 *** Upper Bounds"; Math. Prog. to appear)                                  ***
 ******************************************************************************/
int32_t FletcherAlg2A(
	int32_t Projector, int32_t n, float32_t *vecA, float64_t *b, float64_t c,
	float64_t e, int32_t *iy, float64_t *x, float64_t tol, int32_t *ls,
	int32_t *proj)
{
  int32_t i, j, iter, it, it2, luv, info, lscount = 0, projcount = 0;
  float64_t gd, max, ak, bk, akold, bkold, lamnew, alpha, kktlam, lam_ext;
  float64_t eps     = 1.0e-16;
  float64_t DELTAsv, ProdDELTAsv;

  /*** variables for the adaptive nonmonotone linesearch ***/
  int32_t L, llast;
  float64_t fr, fbest, fv, fc, fv0;

  /*** allocation-dependant settings ***/
#ifdef VARIABLES_ON_STACK

  int32_t ipt[in], ipt2[in], uv[in];
  float64_t g[in], y[in], tempv[in], d[in], Ad[in], t[in],
         xplus[in], tplus[in], sk[in], yk[in];
#else

  int32_t *ipt, *ipt2, *uv;
  float64_t *g, *y, *tempv, *d, *Ad, *t, *xplus, *tplus, *sk, *yk;

  /*** arrays allocation ***/
  ipt   = SG_MALLOC(int32_t, n);
  ipt2  = SG_MALLOC(int32_t, n);
  uv    = SG_MALLOC(int32_t, n);
  g     = SG_MALLOC(float64_t, n);
  y     = SG_MALLOC(float64_t, n);
  tempv = SG_MALLOC(float64_t, n);
  d     = SG_MALLOC(float64_t, n);
  Ad    = SG_MALLOC(float64_t, n);
  t     = SG_MALLOC(float64_t, n);
  xplus = SG_MALLOC(float64_t, n);
  tplus = SG_MALLOC(float64_t, n);
  sk    = SG_MALLOC(float64_t, n);
  yk    = SG_MALLOC(float64_t, n);

#endif

  DELTAsv = EPS_SV * c;
  if (tol <= 1.0e-5 || n <= 20)
      ProdDELTAsv = 0.0F;
  else
      ProdDELTAsv = EPS_SV * c;

  for (i = 0; i < n; i++)
      tempv[i] = -x[i];

  lam_ext = 0.0;

  projcount += InnerProjector(Projector, n, iy, e, tempv, 0, c, x, lam_ext);

  // g = A*x + b;
  // SparseProd(n, t, A, x, ipt);
  {
    float32_t *tempA;

    it = 0;
    for (i = 0; i < n; i++)
        if (fabs(x[i]) > ProdDELTAsv)
            ipt[it++] = i;

    memset(t, 0, n*sizeof(float64_t));
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
    y[i] = g[i] - x[i];
  }

  projcount += InnerProjector(Projector, n, iy, e, y, 0, c, tempv, lam_ext);

  max = alpha_min;
  for (i = 0; i < n; i++)
  {
      y[i] = tempv[i] - x[i];
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
          tempv[i] = alpha*g[i] - x[i];

      projcount += InnerProjector(Projector, n, iy, e, tempv, 0, c, y, lam_ext);

      gd = 0.0;
      for (i = 0; i < n; i++)
      {
          d[i] = y[i] - x[i];
          gd  += d[i] * g[i];
      }

      /* compute Ad = A*d  or  Ad = A*y - t depending on their sparsity */
      {
         float32_t *tempA;

         it = it2 = 0;
         for (i = 0; i < n; i++)
             if (fabs(d[i]) > (ProdDELTAsv*1.0e-2))
                 ipt[it++]   = i;
         for (i = 0; i < n; i++)
             if (fabs(y[i]) > ProdDELTAsv)
                 ipt2[it2++] = i;

         memset(Ad, 0, n*sizeof(float64_t));
         if (it < it2) // compute Ad = A*d
         {
            for (i = 0; i < it; i++)
            {
                tempA = vecA + ipt[i]*n;
                for (j = 0; j < n; j++)
                    Ad[j] += (tempA[j] * d[ipt[i]]);
            }
         }
         else          // compute Ad = A*y-t
         {
            for (i = 0; i < it2; i++)
            {
                tempA = vecA + ipt2[i]*n;
                for (j = 0; j < n; j++)
                    Ad[j] += (tempA[j] * y[ipt2[i]]);
            }
            for (j = 0; j < n; j++)
                Ad[j] -= t[j];
         }
      }

      ak = 0.0;
      for (i = 0; i < n; i++)
          ak += d[i] * d[i];

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
          ak += sk[i] * sk[i];
          bk += sk[i] * yk[i];
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
              if ((x[i] > DELTAsv) && (x[i] < c-DELTAsv))
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
              info   = 1;
              for (i = 0; i < it; i++)
                  if ( fabs(iy[ipt[i]] * g[ipt[i]] + kktlam) > tol )
                  {
                      info = 0;
                      break;
                  }

              if (info == 1)
			  {
                  for (i = 0; i < luv; i++)
				  {
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
				  }
			  }

              if (info == 1)
                  goto Clean;
          }
      }
  }

  SG_SWARNING("Dai-Fletcher method exits after maxvpm = %d iterations.\n", maxvpm)

Clean:

#ifndef VARIABLES_ON_STACK
  SG_FREE(sk);
  SG_FREE(yk);
  SG_FREE(tplus);
  SG_FREE(xplus);
  SG_FREE(t);
  SG_FREE(uv);
  SG_FREE(ipt2);
  SG_FREE(ipt);
  SG_FREE(g);
  SG_FREE(y);
  SG_FREE(tempv);
  SG_FREE(d);
  SG_FREE(Ad);
#endif

  if (ls != NULL)   *ls   = lscount;
  if (proj != NULL) *proj = projcount;
  return(iter);

}

/******************************************************************************/
/*** Encapsulating method to call the chosen Gradient Projection Method     ***/
/******************************************************************************/
int32_t gpm_solver(
	int32_t Solver, int32_t Projector, int32_t n, float32_t *A, float64_t *b,
	float64_t c, float64_t e, int32_t *iy, float64_t *x, float64_t tol,
	int32_t *ls, int32_t *proj)
{
  /*** Uncomment the following if you need to scale the QP Hessian matrix
   *** before calling the chosen solver
  int32_t    i, j;
  float32_t  *ptrA;
  float64_t max, s;

  max = fabs(A[0][0]);
  for (i = 1; i < n; i++)
      if (fabs(A[i][i]) > max)
          max = fabs(A[i][i]);

  s    = 1.0 / max;
  ptrA = vecA;
  for (i = 0; i < n; i++)
      for (j = 0;j < n; j++)
          *ptrA++ = (float32_t)(A[i][j]*s);

  if (Solver == SOLVER_FLETCHER)
      j = FletcherAlg2A(n, vecA, b, c/s, e/s, iy, x, tol, ls);
  else
      j = gvpm(n, vecA, b, c/s, e/s, iy, x, tol, ls);

  for (i = 0; i < n; i++)
      x[i] *= s;
  ***/

  /*** calling the chosen solver with unscaled data ***/
  if (Solver == SOLVER_FLETCHER)
    return FletcherAlg2A(Projector, n, A, b, c, e, iy, x, tol, ls, proj);
  else
    return gvpm(Projector, n, A, b, c, e, iy, x, tol, ls, proj);
}

/******************************************************************************
 *** Piecewise linear monotone target function for the Dai-Fletcher         ***
 *** projector (Y. Dai and R. Fletcher, "New Algorithms for Singly Linear   ***
 *** Constrained Quadratic Programs Subject to Lower and Upper Bounds";     ***
 *** Math. Prog. to appear)                                                 ***
 ******************************************************************************/
float64_t ProjectR(
	float64_t *x, int32_t n, float64_t lambda, int32_t *a, float64_t b,
	float64_t *c, float64_t l, float64_t u)
{
  int32_t i;
  float64_t r = 0.0;

  for (i = 0; i < n; i++)
  {
      x[i] = -c[i] + lambda*(float64_t)a[i];
      if (x[i] >= u) x[i] = u;
      else if (x[i] < l) x[i] = l;
      r += (float64_t)a[i]*x[i];
  }

  return (r - b);
}

/******************************************************************************
 *** Dai-Fletcher QP projector (Y. Dai and R. Fletcher, "New Algorithms for ***
 *** Singly Linear Constrained Quadratic Programs Subject to Lower and      ***
 *** Upper Bounds"; Math. Prog. to appear)                                  ***
 ******************************************************************************/
/***                                                                        ***
 *** Solves the problem        min  x'*x/2 + c'*x                           ***
 ***                       subj to  a'*x - b = 0                            ***
 ***                                l <= x <= u                             ***
 ******************************************************************************/
int32_t ProjectDai(
	int32_t n, int32_t *a, float64_t b, float64_t *c, float64_t l, float64_t u,
	float64_t *x, float64_t &lam_ext)
{
  float64_t lambda, lambdal, lambdau, dlambda, lambda_new, tol_lam;
  float64_t r, rl, ru, s, tol_r;
  int32_t iter;

  tol_lam = 1.0e-11;
  tol_r   = 1.0e-10 * sqrt((u-l)*(float64_t)n);
  lambda  = lam_ext;
  dlambda = 0.5;
  iter    = 1;
  b       = -b;

  // Bracketing Phase
  r = ProjectR(x, n, lambda, a, b, c, l, u);
  if (fabs(r) < tol_r)
      return 0;

  if (r < 0.0)
  {
      lambdal = lambda;
      rl      = r;
      lambda  = lambda + dlambda;
      r       = ProjectR(x, n, lambda, a, b, c, l, u);
      while (r < 0.0)
      {
         lambdal = lambda;
         s       = rl/r - 1.0;
         if (s < 0.1) s = 0.1;
         dlambda = dlambda + dlambda/s;
         lambda  = lambda + dlambda;
         rl      = r;
         r       = ProjectR(x, n, lambda, a, b, c, l, u);
      }
      lambdau = lambda;
      ru      = r;
  }
  else
  {
      lambdau = lambda;
      ru      = r;
      lambda  = lambda - dlambda;
      r       = ProjectR(x, n, lambda, a, b, c, l, u);
      while (r > 0.0)
      {
         lambdau = lambda;
         s       = ru/r - 1.0;
         if (s < 0.1) s = 0.1;
         dlambda = dlambda + dlambda/s;
         lambda  = lambda - dlambda;
         ru      = r;
         r       = ProjectR(x, n, lambda, a, b, c, l, u);
      }
    lambdal = lambda;
    rl      = r;
  }


  // Secant Phase
  s       = 1.0 - rl/ru;
  dlambda = dlambda/s;
  lambda  = lambdau - dlambda;
  r       = ProjectR(x, n, lambda, a, b, c, l, u);

  while (   fabs(r) > tol_r
         && dlambda > tol_lam * (1.0 + fabs(lambda))
         && iter    < maxprojections                )
  {
     iter++;
     if (r > 0.0)
     {
         if (s <= 2.0)
         {
             lambdau = lambda;
             ru      = r;
             s       = 1.0 - rl/ru;
             dlambda = (lambdau - lambdal) / s;
             lambda  = lambdau - dlambda;
         }
         else
         {
             s          = ru/r-1.0;
             if (s < 0.1) s = 0.1;
             dlambda    = (lambdau - lambda) / s;
             lambda_new = 0.75*lambdal + 0.25*lambda;
             if (lambda_new < (lambda - dlambda))
                 lambda_new = lambda - dlambda;
             lambdau    = lambda;
             ru         = r;
             lambda     = lambda_new;
             s          = (lambdau - lambdal) / (lambdau - lambda);
         }
     }
     else
     {
         if (s >= 2.0)
         {
             lambdal = lambda;
             rl      = r;
             s       = 1.0 - rl/ru;
             dlambda = (lambdau - lambdal) / s;
             lambda  = lambdau - dlambda;
         }
         else
         {
             s          = rl/r - 1.0;
             if (s < 0.1) s = 0.1;
             dlambda    = (lambda-lambdal) / s;
             lambda_new = 0.75*lambdau + 0.25*lambda;
             if (lambda_new > (lambda + dlambda))
                 lambda_new = lambda + dlambda;
             lambdal    = lambda;
             rl         = r;
             lambda     = lambda_new;
             s          = (lambdau - lambdal) / (lambdau-lambda);
         }
     }
     r = ProjectR(x, n, lambda, a, b, c, l, u);
  }

  lam_ext = lambda;
  if (iter >= maxprojections)
      SG_SERROR("Projector exits after max iterations: %d\n", iter)

  return (iter);
}

#define SWAP(a,b) { register float64_t t=(a);(a)=(b);(b)=t; }

/*** Median computation using Quick Select ***/
float64_t quick_select(float64_t *arr, int32_t n)
{
  int32_t low, high;
  int32_t median;
  int32_t middle, l, h;

  low    = 0;
  high   = n-1;
  median = (low + high) / 2;

  for (;;)
  {
    if (high <= low)
        return arr[median];

    if (high == low + 1)
    {
        if (arr[low] > arr[high])
            SWAP(arr[low], arr[high]);
        return arr[median];
    }

    middle = (low + high) / 2;
    if (arr[middle] > arr[high]) SWAP(arr[middle], arr[high]);
    if (arr[low]    > arr[high]) SWAP(arr[low],    arr[high]);
    if (arr[middle] > arr[low])  SWAP(arr[middle], arr[low]);

    SWAP(arr[middle], arr[low+1]);

    l = low + 1;
    h = high;
    for (;;)
    {
      do l++; while (arr[low] > arr[l]);
      do h--; while (arr[h]   > arr[low]);
      if (h < l)
          break;
      SWAP(arr[l], arr[h]);
    }

    SWAP(arr[low], arr[h]);
    if (h <= median)
        low = l;
    if (h >= median)
        high = h - 1;
  }
}

/******************************************************************************
 *** Pardalos-Kovoor projector (P.M. Pardalos and N. Kovoor, "An Algorithm  ***
 *** for a Singly Constrained Class of Quadratic Programs Subject to Upper  ***
 *** and Lower Bounds"; Math. Prog. 46, 1990, 321-328).                     ***
 ******************************************************************************
 *** Solves the problem                                                     ***
 ***                       min  x'*x/2 + qk'*x                              ***
 ***                   subj to  iy'*x + e = 0                               ***
 ***                            l <= x <= u                                 ***
 ***                            iy(i) ~= 0                                  ***
 ******************************************************************************/

int32_t Pardalos(
	int32_t n, int32_t *iy, float64_t e, float64_t *qk, float64_t low,
	float64_t up, float64_t *x)
{
  int32_t i, l, iter; /* conters    */
  int32_t luv, lxint; /* dimensions */
  float64_t d, xmin, xmax, xmold, xmid, xx, ts, sw, s, s1, testsum;

  /*** allocation-dependant settings ***/
#ifdef VARIABLES_ON_STACK
  int32_t uv[in], uvt[in];
  float64_t xint[2*in+2], xint2[2*in+2], a[in], b[in], at[in], bt[in];
  float64_t newdia[in], newdt[in];
#else

  int32_t *uv, *uvt;
  float64_t *xint, *xint2, *a, *b, *at, *bt, *newdia, *newdt;

  /*** arrays allocation ***/
  uv     = SG_MALLOC(int32_t, n);
  uvt    = SG_MALLOC(int32_t, n);
  a      = SG_MALLOC(float64_t, n);
  b      = SG_MALLOC(float64_t, n);
  at     = SG_MALLOC(float64_t, n);
  bt     = SG_MALLOC(float64_t, n);
  newdia = SG_MALLOC(float64_t, n);
  newdt  = SG_MALLOC(float64_t, n);
  xint   = SG_MALLOC(float64_t, (2*n + 2));
  xint2  = SG_MALLOC(float64_t, (2*n + 2));

#endif

  d = 0.0;
  for (i = 0; i < n; i++)
      d += iy[i] * qk[i];
  d = 0.5 * (d-e);

  for (i = 0; i < n; i++)
  {
      /* The following computations should divide by iy[i] instead           *
       * of multiply by iy[i], but this is correct for binary classification *
       * with labels -1 and 1.                                               */
      if (iy[i] > 0)
      {
          a[i] = ((qk[i] + low) * iy[i]) * 0.5;
          b[i] = ((up + qk[i]) * iy[i]) * 0.5;
      }
      else
      {
          b[i] = ((qk[i] + low) * iy[i]) * 0.5;
          a[i] = ((up + qk[i]) * iy[i]) * 0.5;
      }
      newdia[i] = (iy[i]*iy[i]);
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

  iter = 0;
  do {
     for (i = 0; i < luv; i++)
     {
         at[i]    = a[uv[i]];
         bt[i]    = b[uv[i]];
         newdt[i] = newdia[uv[i]];
     }

     xmold = xmid;
     xmid = quick_select(xint, lxint);
     if (xmold == xmid)
         xmid = xint[(int32_t)(ThRandPos % lxint)];

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
     memcpy(xint, xint2, lxint*sizeof(float64_t));

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
    memcpy(uv, uvt, luv*sizeof(int32_t));
    iter++;
  } while(luv != 0 && iter < maxprojections);

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
          SG_SWARNING("Inner solver troubles...\n")
  }

  for (i = 0; i < n; i++)
      x[i] = (2.0*x[i]*iy[i]-qk[i]);

#ifndef VARIABLES_ON_STACK
  SG_FREE(newdt);
  SG_FREE(newdia);
  SG_FREE(a);
  SG_FREE(b);
  SG_FREE(uvt);
  SG_FREE(uv);
  SG_FREE(bt);
  SG_FREE(at);
  SG_FREE(xint2);
  SG_FREE(xint);
#endif

  return(iter);
}

/******************************************************************************/
/*** Wrapper method to call the selected inner projector                    ***/
/******************************************************************************/
int32_t InnerProjector(
	int32_t method, int32_t n, int32_t *iy, float64_t e, float64_t *qk,
	float64_t l, float64_t u, float64_t *x, float64_t &lambda)
{
  if (method == 0)
      return Pardalos(n, iy, e, qk, l, u, x);
  else
      return ProjectDai(n, iy, e, qk, l, u, x, lambda);
}
}
/******************************************************************************/
/*** End of gpm.cpp file                                                    ***/
/******************************************************************************/
