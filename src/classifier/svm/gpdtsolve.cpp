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
 *** File:     gpdtsolve.cpp                                                ***
 *** Type:     scalar                                                       ***
 *** Version:  0.9 beta                                                     ***
 *** Date:     July 21, 2004                                                ***
 *** Revision: 1                                                            ***
 ***                                                                        ***
 ******************************************************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "classifier/svm/gpdt.h"
#include "lib/io.h"

#define y_in(i)      y[index_in[(i)]]
#define y_out(i)     y[index_out[(i)]]
#define alpha_in(i)  alpha[index_in[(i)]]
#define alpha_out(i) alpha[index_out[(i)]]
#define minfty       -1000000  // minus infinity

unsigned int Randnext = 1;

#define ThRand    (Randnext = Randnext * 1103515245L + 12345L)
#define ThRandPos ((Randnext = Randnext * 1103515245L + 12345L) & 0x7fffffff)

FILE        *fp;
extern char cOutputStream[10000][80];
extern int  nOutputStream;

//#define OutputStream (cOutputStream[nOutputStream++])
#define     OutputStream stdout
/* the following is to quickly select file or console verbosity output */
#define     sprintf fprintf

/* utility routines prototyping */
void quick_si (int    a[], int k);
void quick_s3 (int    a[], int k, int ia[]);
void quick_s2 (double a[], int k, int ia[]);

/******************************************************************************/
/*** Kernel class constructor                                               ***/
/******************************************************************************/
sKernel::sKernel (CKernel* k, int l)
{
  kernel=k;
  ell=l;
  nor   = NULL;
  vaux  = NULL;
  lx    = NULL;
  ix    = NULL;
  x     = NULL;
  IsSubproblem      = 0;
  KernelEvaluations = 0.0;
}

/******************************************************************************/
/*** Set the problem data for kernel evaluation                             ***/
/******************************************************************************/
void sKernel::SetData(float **x_, int **ix_, int *lx_, int _ell, int _dim)
{
  int i, j, k;

  dim  = _dim;
  ell  = _ell;
  nor  = (double *)malloc(ell*sizeof(double));
  vaux = (float  *)malloc(dim*sizeof(float ));
  memset(vaux, 0, dim*sizeof(float));

  IsSubproblem = 0;
  x  = x_;
  ix = ix_;
  lx = lx_;

  // unroll one (sparse) vector
  vauxRow = 0;
  i       = vauxRow;
  for (k = 0; k < lx[i]; k++)
      vaux[ix[i][k]] = x[i][k];

  // compute the squared Euclidean norm of each vector
  for (i = 0; i < ell; i++)
  {
      nor[i] = 0.0;
      for (j = 0; j < lx[i]; j++)
          nor[i] += (double)(x[i][j]*x[i][j]);
  }
}

/******************************************************************************/
/*** Set the subproblem data                                                ***/
/******************************************************************************/
void sKernel::SetSubproblem(sKernel* ker, int len, int *perm)
{
  int k;

  /* arrays allocations */
  nor  = (double *)malloc(len*sizeof(double   ));
  vaux = (float  *)malloc(ker->dim*sizeof(float));
  memset(vaux, 0, ker->dim*sizeof(float));

  lx = (int    *)malloc(len * sizeof(int    ));
  ix = (int   **)malloc(len * sizeof(int   *));
  x  = (float **)malloc(len * sizeof(float *));
  IsSubproblem = 1;

  for (k = 0; k < len; k++)
  {
      x[k]   = ker->x[perm[k]];
      ix[k]  = ker->ix[perm[k]];
      lx[k]  = ker->lx[perm[k]];
      nor[k] = ker->nor[perm[k]];
  }

  // unroll one (sparse) vector
  vauxRow = 0;
  for (k = 0; k < lx[vauxRow]; k++)
      vaux[ix[vauxRow][k]] = x[vauxRow][k];
}

/******************************************************************************/
/*** Set the selected kernel                                                ***/
/******************************************************************************/
void sKernel::SetKernel(int type, double sigma_, double degree_,
                        double normalisation, double cp)
{
  /*** kernel type:                                           *
   *      0 = linear          (Xi*Xj)                         *
   *      1 = polynomial      (c_poly+(Xi*Xj)*norm)^degree    *
   *      2 = gaussian (RBF)   exp{-sigma*|Xi-Xj|^2}        ***/
  ker_type = type;
  sigma    = sigma_;
  norm     = normalisation;
  c_poly   = cp;
  degree   = degree_;

  if (ker_type == 0 || (ker_type == 1 && degree == 1.0))
  {
      kernel_fun = &sKernel::dot;
  }
  else if (ker_type == 1)
  {
      if (degree != (double)(int)degree)
          kernel_fun = &sKernel::k_pow;
      else
      {
          switch((int)degree)
          {
                case 2:  kernel_fun = &sKernel::k_pow2; break;
                case 3:  kernel_fun = &sKernel::k_pow3; break;
                case 4:  kernel_fun = &sKernel::k_pow4; break;
                case 5:  kernel_fun = &sKernel::k_pow5; break;
                case 6:  kernel_fun = &sKernel::k_pow6; break;
                case 7:  kernel_fun = &sKernel::k_pow7; break;
                case 8:  kernel_fun = &sKernel::k_pow8; break;
                case 9:  kernel_fun = &sKernel::k_pow9; break;
                default: kernel_fun = &sKernel::k_pow;  break;
          }
      }
  }
  else
  {
      kernel_fun = &sKernel::k_gauss;
  }
}

/******************************************************************************/
/*** Kernel class destructor                                                ***/
/******************************************************************************/
sKernel::~sKernel()
{
  int i;

  if (nor  != NULL) free(nor);
  if (vaux != NULL) free(vaux);

  if (lx != NULL) free(lx);
  if (ix != NULL)
  {
      if (!IsSubproblem)
          for (i = 0; i < ell; i++)
              free(ix[i]);
      free(ix);
  }
  if (x != NULL)
  {
      if (!IsSubproblem)
          for (i = 0; i < ell; i++)
              free(x[i]);
      free(x);
  }
}

/******************************************************************************/
/*** Custom implementation of sparse scalar product                         ***/
/******************************************************************************/
double sKernel::dot(int i, int j)
{
  /* store in registers for better performance */
  register int    k;
  register double acc;

  int n     = lx[j];
  int *ip   = ix[j];
  float *xp = x[j];

  if (i != vauxRow)
  {
      for (k = 0; k < lx[vauxRow]; k++)
          vaux[ix[vauxRow][k]] = 0.0;
      vauxRow = i;

      for (k = 0; k < lx[i]; k++)
          vaux[ix[i][k]] = x[i][k];
  }

  acc = 0.0;
  for (k = 0; k < n; k++)
      acc += (double)(xp[k] * vaux[ip[k]]);

  return acc;
}


/******************************************************************************/
/*** Evaluation of linear kernel                                            ***/
/******************************************************************************/
double sKernel::k_lin(int i, int j)
{
  return dot(i, j);
}

/******************************************************************************/
/*** Evaluation of Gaussian (RBF) kernel                                    ***/
/******************************************************************************/
double sKernel::k_gauss(int i, int j)
{
  return exp(-(nor[i] + nor[j] - 2.0*dot(i, j)) * sigma);
}

/******************************************************************************/
/*** Evaluation of deg. 1 polynomial kernel                                 ***/
/******************************************************************************/
double sKernel::k_pow(int i, int j)
{
  return pow(c_poly + dot(i, j)*norm, degree);
}

/******************************************************************************/
/*** Evaluation of deg. 2 polynomial kernel                                 ***/
/******************************************************************************/
double sKernel::k_pow2(int i, int j)
{
  double x = c_poly + dot(i, j)*norm;
  return x*x;
}

/******************************************************************************/
/*** Evaluation of deg. 3 polynomial kernel                                 ***/
/******************************************************************************/
double sKernel::k_pow3(int i, int j)
{
  double x = c_poly + dot(i, j)*norm;
  return x*x*x;
}

/******************************************************************************/
/*** Evaluation of deg. 4 polynomial kernel                                 ***/
/******************************************************************************/
double sKernel::k_pow4(int i, int j)
{
  double x = c_poly + dot(i, j)*norm;
  x = x*x;
  return x*x;
}

/******************************************************************************/
/*** Evaluation of deg. 5 polynomial kernel                                 ***/
/******************************************************************************/
double sKernel::k_pow5(int i, int j)
{
  double t, x = c_poly + dot(i, j)*norm;
  t = x*x;
  return t*t*x;
}

/******************************************************************************/
/*** Evaluation of deg. 6 polynomial kernel                                 ***/
/******************************************************************************/
double sKernel::k_pow6(int i, int j)
{
  double x = c_poly + dot(i, j)*norm;
  x = x*x*x;
  return x*x;
}

/******************************************************************************/
/*** Evaluation of deg. 7 polynomial kernel                                 ***/
/******************************************************************************/
double sKernel::k_pow7(int i, int j)
{
  double t, x = c_poly + dot(i, j)*norm;
  t = x*x*x;
  return t*t*x;
}

/******************************************************************************/
/*** Evaluation of deg. 8 polynomial kernel                                 ***/
/******************************************************************************/
double sKernel::k_pow8(int i, int j)
{
  double x = c_poly + dot(i, j)*norm;
  x = x*x;
  x = x*x;
  return x*x;
}

/******************************************************************************/
/*** Evaluation of deg. 9 polynomial kernel                                 ***/
/******************************************************************************/
double sKernel::k_pow9(int i, int j)
{
  double t, x = c_poly + dot(i, j)*norm;
  t = x*x;
  t = t*t;
  return t*t*x;
}


/******************************************************************************/
/*** Custom implementation of a sparse SAXPY operation                      ***/
/******************************************************************************/
void sKernel::Add(double *v, int j, double mul)
{
  register int k;

  int     n = lx[j];
  int   *ip = ix[j];
  float *xp = x[j];

  for (k = 0; k < n; k++)
      v[ip[k]] += mul*(double)xp[k];
}

/******************************************************************************/
/*** Custom implementation of a sparse scalar product                       ***/
/******************************************************************************/
double sKernel::Prod(double *v, int j)
{
  /* store in registers for better performance */
  register int    k;
  register double acc;

  int     n = lx[j];
  int   *ip = ix[j];
  float *xp = x[j];

  acc = 0.0;
  for (k = 0; k < n; k++)
    acc += (double)xp[k] * v[ip[k]];
  return acc;
}


/******************************************************************************/
/*** Class for caching strategy implementation                              ***/
/******************************************************************************/
class sCache
{

public:
  sCache  (sKernel* sk, int Mbyte, int ell);
  ~sCache ();

  cachetype *FillRow (int row, int IsC = 0);
  cachetype *GetRow  (int row);

  int       DivideMP (int *out, int *in, int n);

  /*** Itarations counter ***/
  void Iteration(void) { nit++; }

  /*** Cache size control ***/
  int CheckCycle(void)
  {
    int us;
    cache_entry *pt = first_free->next;

    for (us = 0; pt != first_free; us++, pt = pt->next);
    if (us != maxmw-1)
        return 1;
    else
        return 0;
  }

private:

  struct cache_entry
  {
    int         row;      // unused row
    int         last_access_it;
    cache_entry *prev, *next;
    cachetype   *data;
  };

  sKernel* KER;
  int     maxmw, ell;
  int     nit;

  cache_entry *mw;
  cache_entry *first_free;
  cache_entry **pindmw;    // 0 if unused row
  cachetype   *onerow;

  cachetype   *FindFree(int row, int IsC);
};


/******************************************************************************/
/*** Cache class constructor                                                ***/
/******************************************************************************/
sCache::sCache(sKernel* sk, int Mbyte, int _ell) : KER(sk), ell(_ell)
{
  int i;

  // size in dwords of one cache row
  maxmw = (sizeof(cache_entry) + sizeof(cache_entry *)
          + ell*sizeof(cachetype)) / 4;
  // number of cache rows
  maxmw = Mbyte*1024*(1024/4) / maxmw;

  /* arrays allocation */
  mw     = (cache_entry  *)malloc(maxmw * sizeof(cache_entry));
  pindmw = (cache_entry **)malloc(ell * sizeof(cache_entry *));
  onerow = (cachetype    *)malloc(ell * sizeof(cachetype    ));

  /* arrays initialization */
  for (i = 0; i < maxmw; i++)
  {
      mw[i].prev           = (i == 0 ? &mw[maxmw-1] : &mw[i-1]);
      mw[i].next           = (i == maxmw-1 ? &mw[0] : &mw[i+1]);
      mw[i].data           = (cachetype *)malloc(ell*sizeof(cachetype));
      mw[i].row            = -1;    // unused row
      mw[i].last_access_it = -1;
  }
  for (i = 0; i < ell; i++)
      pindmw[i] = 0;

  first_free = &mw[0];
  nit        = 0;
}

/******************************************************************************/
/*** Cache class destructor                                                 ***/
/******************************************************************************/
sCache::~sCache()
{
  int i;

  for (i = maxmw-1; i >= 0; i--)
      if (mw[i].data) free(mw[i].data);
  if (onerow) free(onerow);
  if (pindmw) free(pindmw);
  if (mw)     free(mw);
}


/******************************************************************************/
/*** Retrieve a cached row                                                  ***/
/******************************************************************************/
cachetype *sCache::GetRow(int row)
{
  cache_entry *c;

  c = pindmw[row];
  if (c == NULL)
      return NULL;

  c->last_access_it = nit;
  if (c == first_free)
  {
      first_free = first_free->next;
  }
  else
  {
      // move "row" as the most recently used.
      c->prev->next    = c->next;
      c->next->prev    = c->prev;
      c->next          = first_free;
      c->prev          = first_free->prev;
      first_free->prev = c;
      c->prev->next    = c;
  }

  return c->data;
}

/******************************************************************************
 *** Look for a free cache row                                              ***
 *** IMPORTANT: call this method only if you are sure that "row"            ***
 ***            is not already in the cache ( i.e. after calling GetRow() ) ***
 ******************************************************************************/
cachetype *sCache::FindFree(int row, int IsC)
{
  cachetype *pt;

  if (first_free->row != -1) // cache row already contains data
  {
      if (first_free->last_access_it == nit || IsC)
          return 0;
      else
          pindmw[first_free->row] = 0;
  }
  first_free->row            = row;
  first_free->last_access_it = nit;
  pindmw[row]                = first_free;

  pt         = first_free->data;
  first_free = first_free->next;

  return pt;
}


/******************************************************************************/
/*** Enter data in a cache row                                              ***/
/******************************************************************************/
cachetype *sCache::FillRow(int row, int IsC)
{
  int       j;
  cachetype *pt;

  pt = GetRow(row);
  if (pt != NULL)
      return pt;

  pt = FindFree(row, IsC);
  if (pt == 0)
      pt = onerow;

  // Compute all the row elements
  for (j = 0; j < ell; j++)
      pt[j] = (cachetype)KER->Get(row, j);
  return pt;
}


/******************************************************************************/
/*** Expand a sparse row in a full cache row                                ***/
/******************************************************************************/
int sCache::DivideMP(int *out, int *in, int n)
{
   /********************************************************************
    * Input meaning:                                                   *
    *   in  = vector containing row to be extracted in the cache       *
    *   n   = size of in                                               *
    *   out = the indexes of "in" of the components to be computed     *
    *         by this processor (first those in the cache, then the    *
    *         ones not yet computed)                                   *
    * Returns: the number of components of this processor              *
    ********************************************************************/

  int *remained, nremained, k;
  int i;

  remained = (int *)malloc(n*sizeof(int));

  nremained = 0;
  k = 0;
  for (i = 0; i < n; i++)
  {
      if (pindmw[in[i]] != NULL)
          out[k++] = i;
      else
          remained[nremained++] = i;
  }
  for (i = 0; i < nremained; i++)
      out[k++] = remained[i];

  free(remained);
  return n;
}


/******************************************************************************/
/*** Check solution optimality                                              ***/
/******************************************************************************/
int QPproblem::optimal()
{
  /***********************************************************************
   * Returns 1 if the computed solution is optimal, otherwise returns 0. *
   * To verify the optimality it checks (l - chunk_size) training set    *
   * elements: if some sample violates the KKT optimality conditions     *
   * then updates both index_in and index_out and construct the new      *
   * working set.                                                        *
   ***********************************************************************/
  register int i, j, margin_sv_number, z, k, s, kin, z1, znew=0, nnew;

  double gx_i, aux, s1, s2;

  /* sort -y*grad and store in ing the swaps done */
  for (j = 0; j < ell; j++)
  {
      grad[j] = y[j] - st[j];
      ing[j]  = j;
  }

  quick_s2(grad,ell,ing);

  /* compute bee */
  margin_sv_number = 0;

  for (i = chunk_size - 1; i >= 0; i--)
      if (is_free(index_in[i]))
      {
          margin_sv_number++;
          bee = y_in(i) - st[index_in[i]];
          break;
      }

  if (margin_sv_number > 0)
  {
      aux=-1.0;
      for (i = nb-1; i >= 0; i--)
      {
          gx_i = bee + st[index_out[i]];
          if ((is_zero(index_out[i]) && (gx_i*y_out(i) < (1.0-delta))) ||
             (is_bound(index_out[i]) && (gx_i*y_out(i) > (1.0+delta))) ||
             (is_free(index_out[i])  &&
             ((gx_i*y_out(i) < 1.0-delta) || (gx_i*y_out(i) > 1.0+delta))))
          {
              if (fabs(gx_i*y_out(i) - 1.0) > aux)
                  aux = fabs(gx_i*y_out(i) - 1.0);
          }
      }
  }
  else
  {
      for (i = ell - 1; i >= 0; i--)
          if (is_free(i))
          {
              margin_sv_number++;
              bee = y[i] - st[i];
              break;
          }
      if (margin_sv_number == 0)
      {
          s1 = -minfty;
          s2 = -minfty;
          for (j = 0; j < ell; j++)
              if ( (alpha[ing[j]] > DELTAsv) &&  (y[ing[j]] >= 0) )
              {
                  s1 = grad[j];
                  break;
              }
          for (j = 0; j < ell; j++)
              if ( (alpha[ing[j]] < c_const-DELTAsv) && (y[ing[j]] <= 0) )
              {
                  s2 = grad[j];
                  break;
              }
          if (s1 < s2)
              aux = s1;
          else
              aux = s2;

          s1 = minfty;
          s2 = minfty;
          for (j = ell-1; j >=0; j--)
              if ( (alpha[ing[j]] > DELTAsv) && (y[ing[j]] <= 0) )
              {
                  s1 = grad[j];
                  break;
              }
          for (j = ell-1; j >=0; j--)
              if ( (alpha[ing[j]] < c_const-DELTAsv) && (y[ing[j]] >= 0) )
              {
                  s2 = grad[j];
                  break;
              }
          if (s2 > s1) s1 = s2;

          bee = 0.5 * (s1+aux);
      }

      aux = -1.0;
      for (i = ell-1; i >= 0; i--)
      {
          gx_i = bee + st[i];
          if ((is_zero(i) && (gx_i*y[i] < (1.0-delta))) ||
             (is_bound(i) && (gx_i*y[i] > (1.0+delta))) ||
             (is_free(i)  &&
             ((gx_i*y[i] < 1.0-delta) || (gx_i*y[i] > 1.0+delta))))
          {
              if (fabs(gx_i*y[i] - 1.0) > aux)
                  aux = fabs(gx_i*y[i] - 1.0);
          }
      }
  }

  if (aux < 0.0)
      return 1;
  else
  {
      if (verbosity > 1)
          sprintf(OutputStream, "  Max KKT violation: %lf\n", aux);
      else if (verbosity > 0)
          sprintf(OutputStream, "  %lf\n", aux);

      if (fabs(kktold-aux) < delta*0.01 &&  aux < delta*2.0)
      {
          if (DELTAvpm > InitialDELTAvpm*0.1)
          {
              DELTAvpm = (DELTAvpm*0.5 > InitialDELTAvpm*0.1 ?
                                            DELTAvpm*0.5 : InitialDELTAvpm*0.1);
              sprintf(OutputStream,
                      "Inner tolerance changed to: %lf\n", DELTAvpm);
          }
      }

      kktold = aux;

      for (j = 0; j < chunk_size; j++)
          inold[j] = index_in[j];

      z  = -1;  /* index of the last entered component from the top    */
      z1 = ell; /* index of the last entered component from the bottom */
      k  = 0;
      j  = 0;
      while (k < q)
      {
          i = z + 1; /* index of the candidate components from the top */
          while (i < z1)
          {
              if ( is_free(ing[i]) ||
                   (-y[ing[i]]>=0 && is_zero(ing[i])) ||
                   (-y[ing[i]]<=0 && is_bound(ing[i]))
                 )
              {
                  znew = i; /* index of the component to select from the top */
                  break;
              }
              i++;
          }
          if (i == z1)
              break;

          s = z1 - 1;
          while (znew < s)
          {
              if ( is_free(ing[s]) ||
                   (y[ing[s]]>=0 && is_zero(ing[s])) ||
                   (y[ing[s]]<=0 && is_bound(ing[s]))
                 )
              {
                  z1 = s;
                  z  = znew;
                  break;
              }
              s--;
          }
          if (znew == s)
          break;

          index_in[k++] = ing[z];
          index_in[k++] = ing[z1];
      }

      if (k < q)
      {
          if (verbosity > 1)
              sprintf(OutputStream, "  New q: %i\n", k);
          q = k;
      }

      quick_si(index_in, q);

      s = 0;
      j = 0;
      for (k = 0; k < chunk_size; k++)
      {
          z = inold[k];
          for (i = j; i < q; i++)
              if (z <= index_in[i])
                  break;

          if (i == q)
          {
              for (i = k; i < chunk_size; i++)
              {
                  ing[s] = inold[i]; /* older components not in the new basis */
                  s      = s+1;
              }
              break;
          }

          if (z == index_in[i])
              j      = i + 1; /* the component is still in the basis  */
          else
          {
              ing[s] = z;     /* older component not in the new basis */
              s      = s + 1;
              j      = i;
          }
      }

      for (i = 0; i < s; i++)
      {
          bmemrid[i] = bmem[ing[i]];
          pbmr[i]    = i;
      }

      quick_s3(bmemrid, s, pbmr);

      /* check if support vectors not at bound enter the basis */
      j = q;
      i = 0;
      while (j < chunk_size && i < s)
      {
          if (is_free(ing[pbmr[i]]))
          {
              index_in[j] = ing[pbmr[i]];
              j++;
          }
          i++;
      }

      /* choose which bound variables keep in basis (alpha = 0 or alpha = C) */
      if (j < chunk_size)
      {
          i = 0;
          while (j < chunk_size && i < s)
          {
              if (is_zero(ing[pbmr[i]]))
              {
                  index_in[j] = ing[pbmr[i]];
                  j++;
              }
              i++;
          }
          if (j < chunk_size)
          {
              i = 0;
              while (j < chunk_size && i < s)
              {
                  if (is_bound(ing[pbmr[i]]))
                  {
                      index_in[j] = ing[pbmr[i]];
                      j++;
                  }
                  i++;
              }
          }
      }

      quick_si(index_in, chunk_size);

      for (i = 0; i < chunk_size; i++)
          bmem[index_in[i]]++;

      j = 0;
      k = 0;
      for (i = 0; i < chunk_size; i++)
      {
          for (z = j; z < index_in[i]; z++)
          {
              index_out[k] = z;
              k++;
          }
          j = index_in[i]+1;
      }
      for (z = j; z < ell; z++)
      {
          index_out[k] = z;
          k++;
      }

      for (i = 0; i < nb; i++)
          bmem[index_out[i]] = 0;

      kin = 0;
      j   = 0;
      for (k = 0; k < chunk_size; k++)
      {
          z = index_in[k];
          for (i = j; i < chunk_size; i++)
              if (z <= inold[i])
                  break;
          if (i == chunk_size)
          {
              for (s = k; s < chunk_size; s++)
              {
                  incom[s] = -1;
                  cec[index_in[s]]++;
              }
              kin = kin + chunk_size - k ;
              break;
          }

          if (z == inold[i])
          {
              incom[k] = i;
              j        = i+1;
          }
          else
          {
              incom[k] = -1;
              j        = i;
              kin      = kin + 1;
              cec[index_in[k]]++;
          }
      }

      nnew = kin & (~1);
      if (nnew < 10)
          nnew = 10;
      if (nnew < chunk_size/10)
          nnew = chunk_size/10;
      if (nnew < q)
      {
          q = nnew;
          q = q & (~1);
      }

      if (verbosity > 1)
          sprintf(OutputStream,
          "  Working set: new components: %i,  new parameter n: %i\n", kin, q);

      return 0;
   }
}

/******************************************************************************/
/*** Optional preprocessing: random distribution                            ***/
/******************************************************************************/
int QPproblem::Preprocess0(int *aux, int *sv)
{
  int i, j;

  Randnext = 1;
  memset(sv, 0, ell*sizeof(int));
  for (i = 0; i < chunk_size; i++)
  {
      do
      {
          j = ThRandPos % ell;
      } while (sv[j] != 0);
      sv[j] = 1;
  }
  return(chunk_size);
}

/******************************************************************************/
/*** Optional preprocessing: block parallel distribution                    ***/
/******************************************************************************/
int QPproblem::Preprocess1(sKernel* KER, int *aux, int *sv)
{
  int    s;    // elements owned by the processor
  int    sl;   // elements of the n-1 subproblems
  int    n, i, off, j, k, ll;
  int    nsv, nbsv;
  int    *sv_loc, *bsv_loc, *sp_y;
  float  *sp_D=NULL;
  double *sp_alpha, *sp_h;

  s  = ell;
  /* divide the s elements into n blocks smaller than preprocess_size */
  n  = (s + preprocess_size - 1) / preprocess_size;
  sl = 1 + s / n;

  if (verbosity > 0)
      sprintf(OutputStream,
          "  Preprocessing: examples = %d, subp. = %d, size = %d\n",s,n,sl);

  sv_loc   = (int    *)malloc(s*sizeof(int    ));
  bsv_loc  = (int    *)malloc(s*sizeof(int    ));
  sp_alpha = (double *)malloc(sl*sizeof(double));
  sp_h     = (double *)malloc(sl*sizeof(double));
  sp_y     = (int    *)malloc(sl*sizeof(int   ));

  if (sl < 500)
      sp_D = (float *)malloc(sl*sl * sizeof(float));
  else
	  CIO::message(M_ERROR,"sl<500 expect me to die\n");

  for (i = 0; i < sl; i++)
       sp_h[i] = -1.0;
  memset(alpha, 0, ell*sizeof(double));

  /* randomly reorder the component to select */
  for (i = 0; i < ell; i++)
      aux[i] = i;
  Randnext = 1;
  for (i = 0; i < ell; i++)
  {
      j  = ThRandPos % ell;
      k  = ThRandPos % ell;
      ll = aux[j]; aux[j] = aux[k]; aux[k] = ll;
  }

  nbsv = nsv = 0;
  for (i = 0; i < n; i++)
  {
      if (verbosity > 0)
          sprintf(OutputStream, "%d...", i);
      SplitParts(s, i, n, &ll, &off);

      if (sl < 500)
      {
          for (j = 0; j < ll; j++)
          {
              sp_y[j] = y[aux[j+off]];
              for (k = j; k < ll; k++)
                  sp_D[k*sl + j] = sp_D[j*sl + k]
                                 = y[aux[j+off]] * y[aux[k+off]]
                                   * (float)KER->Get(aux[j+off], aux[k+off]);
          }

          memset(sp_alpha, 0, sl*sizeof(double));

          /* call the gradient projection QP solver */
          gpm_solver(projection_solver, ll, sp_D, sp_h, c_const, 0.0,
                     sp_y, sp_alpha, delta*10, NULL);
      }
      else
      {
          QPproblem p2;
          p2.Subproblem(*this, ll, aux + off);
          p2.chunk_size     = (int) ( (double)chunk_size / sqrt((double)n) );
          p2.q              = (int) ( (double)q / sqrt((double)n) );
          p2.maxmw          = ll*ll*4 / (1024 * 1024);
          if (p2.maxmw > maxmw / 2)
              p2.maxmw = maxmw / 2;
          p2.verbosity      = 0;
          p2.delta          = delta * 10.0;
          p2.PreprocessMode = 0;
          KER->KernelEvaluations += p2.gpdtsolve(sp_alpha);
      }

      for (j = 0; j < ll; j++)
      {
          /* modify bound support vector approximation */
          if (sp_alpha[j] < (c_const-DELTAsv))
              sp_alpha[j] = 0.0;
          else
              sp_alpha[j] = c_const;
          if (sp_alpha[j] > DELTAsv)
          {
              if (sp_alpha[j] < (c_const-DELTAsv))
                  sv_loc[nsv++]   = aux[j+off];
              else
                  bsv_loc[nbsv++] = aux[j+off];
              alpha[aux[j+off]] = sp_alpha[j];
          }
      }
  }

  Randnext = 1;

  /* add the known support vectors to the working set */
  memset(sv, 0, ell*sizeof(int));
  ll = (nsv < chunk_size ? nsv : chunk_size);
  for (i = 0; i < ll; i++)
  {
      do {
           j = sv_loc[ThRandPos % nsv];
      } while (sv[j] != 0);
      sv[j] = 1;
  }

  /* add the known bound support vectors to the working set */
  ll = ((nsv+nbsv) < chunk_size ? (nsv+nbsv) : chunk_size);
  for (; i < ll; i++)
  {
      do {
           j = bsv_loc[ThRandPos % nbsv];
      } while (sv[j] != 0);
      sv[j] = 1;
  }

  /* eventually fill up the working set with other components randomly chosen */
  for (; i < chunk_size; i++)
  {
      do {
           j = ThRandPos % ell;
      } while (sv[j] != 0);
      sv[j] = 1;
  }


  /* dealloc temporary arrays */
  if (sl < 500) free(sp_D);
  free(sp_y    );
  free(sp_h    );
  free(sv_loc  );
  free(bsv_loc );
  free(sp_alpha);

  if (verbosity > 0)
      sprintf(OutputStream,
              "\n  Preprocessing: SV = %d, BSV = %d\n", nsv, nbsv);

  return(nsv);
}

/******************************************************************************/
/*** Compute the QP problem solution                                        ***/
/******************************************************************************/
double QPproblem::gpdtsolve(double *solution)
{
  int       i, j, k, z, iin, jin, nit, tot_vpm_iter, lsCount;
  int       nzin, nzout;
  int       *sp_y;               /* labels vector                             */
  int       *indnzin, *indnzout; /* nonzero components indices vectors        */
  float     *sp_D;               /* quadratic part of the objective function  */
  double    *sp_h, *sp_hloc,     /* linear part of the objective function     */
            *sp_alpha,*stloc;    /* variables and gradient updating vectors   */
  double    sp_e, aux, fval;
  double    *vau;
  double    *weight;
  double    tot_prep_time, tot_vpm_time, tot_st_time, total_time;
  sCache    *Cache;
  cachetype *ptmw;
  clock_t   t, ti;

  Cache = new sCache(KER, maxmw, ell);
    if (chunk_size > ell) chunk_size = ell;

  kktold = 10000.0;
  if (delta <= 5e-3)
  {
      if ( (chunk_size <= 20) | ((double)chunk_size/ell <= 0.001) )
          DELTAvpm = delta * 0.1;
      else if ( (chunk_size <= 200) | ((double)chunk_size/ell <= 0.005) )
          DELTAvpm = delta * 0.5;
      else
          DELTAvpm = delta;
  }
  else
  {
      if ( (chunk_size <= 200) | ((double)chunk_size/ell <= 0.005) )
          DELTAvpm = (1e-3 < delta*0.1) ? 1e-3 : delta*0.1;
      else
          DELTAvpm = 5e-3;
  }

  InitialDELTAvpm = DELTAvpm;
  DELTAsv         = EPS_SV * c_const;

  q               = q & (~1);
  nb              = ell - chunk_size;
  tot_vpm_iter    = 0;

  tot_prep_time = tot_vpm_time = tot_st_time = total_time = 0.0;

  ti = clock();

  /* arrays allocation */
  CIO::message(M_DEBUG,"ell:%d, chunk_size:%d, nb:%d dim:%d\n", ell, chunk_size,nb, dim);
  ing       = (int    *)malloc(ell*sizeof(int       ));
  inaux     = (int    *)malloc(ell*sizeof(int       ));
  index_in  = (int    *)malloc(chunk_size*sizeof(int));
  index_out = (int    *)malloc(ell*sizeof(int       ));
  indnzout  = (int    *)malloc(nb*sizeof(int        ));
  alpha     = (double *)malloc(ell*sizeof(double    ));

  memset(alpha, 0, ell*sizeof(double));
  memset(ing,   0, ell*sizeof(int));

  if (verbosity > 0 && PreprocessMode != 0)
      sprintf(OutputStream, "\n*********** Begin setup step...\n");
  t = clock();

  switch(PreprocessMode)
  {
    case 1: Preprocess1(KER, inaux, ing); break;
    case 0:
    default:
            Preprocess0(inaux, ing); break;
  }

  for (j = k = i = 0; i < ell; i++)
      if (ing[i] == 0)
          index_out[j++] = i;
      else
          index_in[k++]  = i;

  t = clock() - t;
  if (verbosity > 0 && PreprocessMode != 0)
  {
      sprintf(OutputStream,
              "  Time for setup: %.2lf\n", (double)t/CLOCKS_PER_SEC);
      sprintf(OutputStream,
              "\n\n*********** Begin decomposition technique...\n");
  }

  /* arrays allocation */
  bmem     = (int    *)malloc(ell*sizeof(int       ));
  bmemrid  = (int    *)malloc(chunk_size*sizeof(int));
  pbmr     = (int    *)malloc(chunk_size*sizeof(int));
  cec      = (int    *)malloc(ell*sizeof(int       ));
  indnzin  = (int    *)malloc(chunk_size*sizeof(int));
  inold    = (int    *)malloc(chunk_size*sizeof(int));
  incom    = (int    *)malloc(chunk_size*sizeof(int));
  vau      = (double *)malloc(ell*sizeof(double    ));
  grad     = (double *)malloc(ell*sizeof(double    ));
  weight   = (double *)malloc(dim*sizeof(double    ));
  st       = (double *)malloc(ell*sizeof(double    ));
  stloc    = (double *)malloc(ell*sizeof(double    ));

  for (i = 0; i < ell; i++)
  {
    bmem[i] = 0;
    cec[i]  = 0;
    st[i]   = 0;
  }

  sp_y     = (int    *)malloc(chunk_size*sizeof(int               ));
  sp_D     = (float  *)malloc(chunk_size*chunk_size * sizeof(float));
  sp_alpha = (double *)malloc(chunk_size*sizeof(double            ));
  sp_h     = (double *)malloc(chunk_size*sizeof(double            ));
  sp_hloc  = (double *)malloc(chunk_size*sizeof(double            ));

  for (i = 0; i < chunk_size; i++)
      cec[index_in[i]] = cec[index_in[i]]+1;

  for (i = chunk_size-1; i >= 0; i--)
  {
      incom[i]          = -1;
      sp_alpha[i]       = 0.0;
      bmem[index_in[i]] = 1;
  }

  if (verbosity == 1)
  {
      sprintf(OutputStream, "  IT  | Prep Time | Solver IT | Solver Time |");
      sprintf(OutputStream, " Grad Time | KKT violation\n");
      sprintf(OutputStream, "------+-----------+-----------+-------------+");
      sprintf(OutputStream, "-----------+--------------\n");
  }

  /***************************************************************************/
  /* Begin the problem resolution loop                                       */
  nit = 0;
  do
  {
      t = clock();
      if ((nit % 10) == 9)
      {
          if ((t-ti) > 0)
              total_time += (double)(t-ti) / CLOCKS_PER_SEC;
          else
              total_time += (double)(ti-t) / CLOCKS_PER_SEC;
          ti = t;
      }

      if (verbosity > 1)
          sprintf(OutputStream, "\n*********** ITERATION: %d\n", nit + 1);
      else if (verbosity > 0)
          sprintf(OutputStream, "%5d |", nit + 1);
      else
          sprintf(OutputStream, ".");
      fflush(stdout);

      nzout = 0;
      for (k = 0; k < nb; k++)
          if (alpha_out(k)>DELTAsv)
          {
              indnzout[nzout] = index_out[k];
              nzout++;
          }

      sp_e = 0.;
      for (j = 0; j < nzout; j++)
      {
          vau[j] = y[indnzout[j]]*alpha[indnzout[j]];
          sp_e  += vau[j];
      }

      if (verbosity > 1)
          sprintf(OutputStream, "  spe: %e ", sp_e);

      for (i = 0; i < chunk_size; i++)
          sp_y[i] = y_in(i);

      /* Construct the objective function Hessian */
      for (i = 0; i < chunk_size; i++)
      {
          iin  = index_in[i];
          ptmw = Cache->GetRow(iin);
          if (ptmw != 0)
          {
              for (j = 0; j <= i; j++)
                  sp_D[i*chunk_size + j] = sp_y[i]*sp_y[j] * ptmw[index_in[j]];
          }
          else if (incom[i] == -1)
              for (j = 0; j <= i; j++)
                  sp_D[i*chunk_size + j] = sp_y[i]*sp_y[j]
                                           * (float)KER->Get(iin, index_in[j]);
          else
          {
              for (j = 0; j < i; j++)
                  if (incom[j] == -1)
                      sp_D[i*chunk_size + j]
                         = sp_y[i]*sp_y[j] * (float)KER->Get(iin, index_in[j]);
                  else
                      sp_D[i*chunk_size + j]
                         = sp_D[incom[j]*chunk_size + incom[i]];
              sp_D[i*chunk_size + i]
                  = sp_y[i]*sp_y[i] * (float)KER->Get(iin, index_in[i]);
          }
      }
      for (i = 0; i < chunk_size; i++)
      {
          for (j = 0; j < i; j++)
              sp_D[j*chunk_size + i] = sp_D[i*chunk_size + j];
      }

      if (nit == 0 && PreprocessMode > 0)
      {
         for (i = 0; i < chunk_size; i++)
         {
             iin  = index_in[i];
             aux  = 0.;
             ptmw = Cache->GetRow(iin);
             if (ptmw == NULL)
                 for (j = 0; j < nzout; j++)
                     aux += vau[j] * KER->Get(iin, indnzout[j]);
             else
                 for (j = 0; j < nzout; j++)
                     aux += vau[j] * ptmw[indnzout[j]];
             sp_h[i] = y_in(i) * aux - 1.0;
         }
      }
      else
      {
         for (i = 0; i < chunk_size; i++)
             vau[i] = alpha_in(i);
         for (i = 0; i < chunk_size; i++)
         {
             aux = 0.0;
             for (j = 0; j < chunk_size; j++)
                 aux += sp_D[i*chunk_size + j] * vau[j];
             sp_h[i] = st[index_in[i]] * y_in(i) - 1.0 - aux;
         }
      }

    /*** Proximal point modification: first type ***/

    aux = fabs(sp_D[0]);
    for (i = 1; i < chunk_size; i++)
        if (fabs(sp_D[i*chunk_size + i]) > aux)
            aux = fabs(sp_D[i*chunk_size + i]);
    for (i = 0; i < chunk_size; i++)
    {
        vau[i]                  = sp_D[i*chunk_size + i];
        sp_h[i]                -= aux* tau_proximal * alpha_in(i);
        sp_D[i*chunk_size + i] += (float)(aux*tau_proximal);
    }

    t = clock() - t;
    if (verbosity > 1)
        sprintf(OutputStream,
                "  Preparation Time: %.2lf\n", (double)t/CLOCKS_PER_SEC);
    else if (verbosity > 0)
        sprintf(OutputStream, "  %8.2lf |", (double)t/CLOCKS_PER_SEC);
    tot_prep_time += (double)t/CLOCKS_PER_SEC;

    t = clock();
    if (kktold < delta*100)
        for (i = 0; i < chunk_size; i++)
            sp_alpha[i] = alpha_in(i);
    else
        for (i = 0; i < chunk_size; i++)
            sp_alpha[i] = 0.0;


    /*** call the chosen inner gradient projection QP solver ***/
    i = gpm_solver(projection_solver, chunk_size, sp_D, sp_h, c_const,
                   sp_e, sp_y, sp_alpha, DELTAvpm, &lsCount);

    t = clock() - t;
    if (verbosity > 1)
        sprintf(OutputStream, "  Solver it: %d, ls: %d, time: %.2lf\n",
                                         i, lsCount, (double)t/CLOCKS_PER_SEC);
    else if (verbosity > 0)
        sprintf(OutputStream, "    %6d |    %8.2lf |",
                                         i, (double)t/CLOCKS_PER_SEC);

    tot_vpm_iter += i;
    tot_vpm_time += (double)t/CLOCKS_PER_SEC;

    /*** Proximal point modification: second type ***/

    for (i = 0; i < chunk_size; i++)
        sp_D[i*chunk_size + i] = (float)vau[i];

    t = clock();

    nzin = 0;
    for (j = 0; j < chunk_size; j++)
    {
        if (nit == 0)
            aux = sp_alpha[j];
        else
            aux = sp_alpha[j] - alpha_in(j);
        if (fabs(aux) > DELTAsv)
        {
            indnzin[nzin] = index_in[j];
            grad[nzin]    = aux * y_in(j);
            nzin++;
        }
    }

    if (ker_type != 0)  // nonlinear kernel
    {
        k = Cache->DivideMP(ing, indnzin, nzin);
        for (j = 0; j < k; j++)
        {
            ptmw = Cache->FillRow(indnzin[ing[j]]);
            for (i = 0; i < ell; i++)
                st[i] += grad[ing[j]] * ptmw[i];
        }

        if (PreprocessMode > 0 && nit == 0)
        {
            clock_t ti2;

            ti2 = clock();
            for (j = 0; j < nzout; j++)
            {
                jin  = indnzout[j];
                ptmw = Cache->FillRow(jin);
                for (i = 0; i < ell; i++)
                    st[i] += alpha[jin] * y[jin] * ptmw[i];
            }
            if (verbosity > 1)
                sprintf(OutputStream,
                 "  G*x0 time: %.2lf\n", (double)(clock()-ti2)/CLOCKS_PER_SEC);
        }
    }
    else                // linear kernel
    {
        for (i = 0; i < dim; i++) weight[i] = 0.0;

        for (j = 0; j < nzin; j++)
            KER->Add(weight, indnzin[j], grad[j]);
        if (nit == 0 && PreprocessMode > 0)
            for (j = 0; j < nzout; j++)
            {
                jin = indnzout[j];
                KER->Add(weight, jin, alpha[jin] * y[jin]);
            }

        for (i = 0; i < ell; i++)
            st[i] += KER->Prod(weight, i);
    }

    /*** sort the vectors for cache managing ***/

    t = clock() - t;
    if (verbosity > 1)
        sprintf(OutputStream,
                "  Gradient updating time: %.2lf\n", (double)t/CLOCKS_PER_SEC);
    else if (verbosity > 0)
        sprintf(OutputStream, "  %8.2lf |", (double)t/CLOCKS_PER_SEC);
    tot_st_time += (double)t/CLOCKS_PER_SEC;

    /* global updating of the solution vector */
    for (i = 0; i < chunk_size; i++)
        alpha_in(i) = sp_alpha[i];

    if (verbosity > 1)
    {
        j = k = 0;
        for (i = 0; i < ell; i++)
        {
            if (is_free(i))  j++;
            if (is_bound(i)) k++;
        }
        sprintf(OutputStream, "  SV: %d,  BSV: %d\n", j+k, k);
    }
    Cache->Iteration();
    nit = nit+1;
  } while (!optimal());
  /* End of the problem resolution loop                                      */
  /***************************************************************************/

  t = clock();
  if ((t-ti) > 0)
      total_time += (double)(t-ti) / CLOCKS_PER_SEC;
  else
      total_time += (double)(ti-t) / CLOCKS_PER_SEC;
  ti = t;

  memcpy(solution, alpha, ell * sizeof(double));

  /* objective function evaluation */
  fval = 0.0;
  for (i = 0; i < ell; i++)
      fval += alpha[i]*(y[i]*st[i]*0.5 - 1.0);

  sprintf(OutputStream, "\n------+-----------+-----------+-------------+");
  sprintf(OutputStream, "-----------+--------------\n");
  sprintf(OutputStream,
      "\n- TOTAL ITERATIONS: %i\n", nit);

  if (verbosity > 1)
  {
      j = 0;
      k = 0;
      z = 0;
      for (i = ell - 1; i >= 0; i--)
      {
           if (cec[i] > 0) j++;
           if (cec[i] > 1) k++;
           if (cec[i] > 2) z++;
      }
      sprintf(OutputStream,
        "- Variables entering the working set at least one time:  %i\n", j);
      sprintf(OutputStream,
        "- Variables entering the working set at least two times:  %i\n", k);
      sprintf(OutputStream,
        "- Variables entering the working set at least three times:  %i\n", z);
  }


  free(bmem);
  free(bmemrid);
  free(pbmr);
  free(cec);
  free(ing);
  free(inaux);
  free(indnzin);
  free(index_in);
  free(inold);
  free(incom);
  free(indnzout);
  free(index_out);
  free(vau);
  free(alpha);
  free(weight);
  free(grad);
  free(stloc);
  free(st);
  free(sp_h);
  free(sp_y);
  free(sp_D);
  free(sp_alpha);
  delete Cache;

  aux = KER->KernelEvaluations;
  sprintf(OutputStream, "- Total CPU time: %lf\n", total_time);
  if (verbosity > 0)
  {
    sprintf(OutputStream, "- Total kernel evaluations: %.0lf\n", aux);
    sprintf(OutputStream, "- Total inner sover iterations: %i\n", tot_vpm_iter);
    sprintf(OutputStream, "- Total preparation time: %lf\n", tot_prep_time);
    sprintf(OutputStream, "- Total inner solver time: %lf\n", tot_vpm_time);
    sprintf(OutputStream, "- Total gradient updating time: %lf\n", tot_st_time);
  }
  sprintf(OutputStream, "- Objective function value: %lf\n", fval);
  return aux;
}

/******************************************************************************/
/*** Write out the computed solution                                        ***/
/******************************************************************************/
void QPproblem::write_solution(FILE *fp, double *sol)
{
  register int i, j;
  int sv_number, bsv;

  bsv = sv_number = 0;
  for (i = 0; i < ell; i++)
      if (sol[i] > DELTAsv)
      {
          sv_number++;
          if (sol[i] > (c_const - DELTAsv)) bsv++;
      }

  /*** write out to model file the same information as SVMlight ***
   *   This is intended to give the user the chance to use the    *
   *   SVMlight classification module 'svm_classify' to make      *
   *   predictions.                                               */
  fprintf(fp, "GPDT model output (same format as Joachims' SVMlight)\n");
  fprintf(fp, "%d # kernel type\n", ker_type);
  fprintf(fp, "%d # kernel parameter -d\n", (int)KER->degree);
  fprintf(fp, "%.8g # kernel parameter -g\n", KER->sigma);
  fprintf(fp, "%.8g # kernel parameter -s\n", KER->norm);
  fprintf(fp, "%.8g # kernel parameter -r\n", KER->c_poly);
  fprintf(fp, "empty# kernel parameter -u\n");
  fprintf(fp, "%d # highest feature index\n", dim-1);
  fprintf(fp, "%d # number of training documents\n", ell);
  fprintf(fp, "%d # number of support vectors plus 1\n", sv_number + 1);
  fprintf(fp, "%.8g # threshold b, each following line is a SV", -bee);
  fprintf(fp, " (starting with alpha*y)\n");

  for (i = 0; i < ell; i++)
      if (sol[i] > DELTAsv)
      {
          fprintf(fp, "%.32g", (double)(sol[i]*y[i]));
          for (j = 0; j < KER->lx[i]; j++)
              fprintf(fp, " %d:%.8g", KER->ix[i][j], (double)KER->x[i][j]);
          fprintf(fp, "\n");
      }
  sprintf(OutputStream, "- SV  = %d\n", sv_number);
  sprintf(OutputStream, "- BSV = %d\n", bsv);
  sprintf(OutputStream, "- Threshold b = %.8g\n", -bee);
}


/******************************************************************************/
/*** Quick sort for integer vectors                                         ***/
/******************************************************************************/
void quick_si(int a[], int n)
{
  int i, j, s, d, l, x, w, ps[20], pd[20];

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
                  w    = a[i];
                  a[i] = a[j];
                  i++;
                  a[j] = w;
                  j--;
              }
          } while(i<=j);
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
          }
      } while (s < d);
  } while (l >= 0);
}

/******************************************************************************/
/*** Quick sort for real vectors returning also the exchanges               ***/
/******************************************************************************/
void quick_s2(double a[], int n, int ia[])
{
  int     i, j, s, d, l, iw, ps[20], pd[20];
  double  x, w;

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
                  iw    = ia[i];
                  w     = a[i];
                  ia[i] = ia[j];
                  a[i]  = a[j];
                  i++;
                  ia[j] = iw;
                  a[j]  = w;
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
          }
      } while (s < d);
  } while(l>=0);
}

/******************************************************************************/
/*** Quick sort for integer vectors returning also the exchanges            ***/
/******************************************************************************/
void quick_s3(int a[], int n, int ia[])
{
  int i, j, s, d, l, iw, w, x, ps[20], pd[20];

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
                 iw    = ia[i];
                 w     = a[i];
                 ia[i] = ia[j];
                 a[i]  = a[j];
                 i++;
                 ia[j] = iw;
                 a[j]  = w;
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
          }
      } while (s < d);
  } while (l >= 0);
}

/******************************************************************************/
/*** End of gpdtsolve.cpp file                                              ***/
/******************************************************************************/
