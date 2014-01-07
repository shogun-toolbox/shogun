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
 *** File:     gpdtsolve.cpp                                                ***
 *** Type:     scalar                                                       ***
 *** Version:  1.0                                                          ***
 *** Date:     November, 2006                                                ***
 *** Revision: 2                                                            ***
 ***                                                                        ***
 *** SHOGUN adaptions  Written (W) 2006-2009 Soeren Sonnenburg              ***
 ******************************************************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <lib/external/gpm.h>
#include <lib/external/gpdt.h>
#include <lib/external/gpdtsolve.h>
#include <lib/Signal.h>
#include <io/SGIO.h>

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace shogun
{
#define y_in(i)      y[index_in[(i)]]
#define y_out(i)     y[index_out[(i)]]
#define alpha_in(i)  alpha[index_in[(i)]]
#define alpha_out(i) alpha[index_out[(i)]]
#define minfty       (-1.0e+30)  // minus infinity

uint32_t Randnext = 1;

#define ThRand    (Randnext = Randnext * 1103515245L + 12345L)
#define ThRandPos ((Randnext = Randnext * 1103515245L + 12345L) & 0x7fffffff)

FILE        *fp;

/* utility routines prototyping */
void quick_si (int32_t a[], int32_t k);
void quick_s3 (int32_t a[], int32_t k, int32_t ia[]);
void quick_s2 (float64_t a[], int32_t k, int32_t ia[]);

/******************************************************************************/
/*** Class for caching strategy implementation                              ***/
/******************************************************************************/
class sCache
{

public:
  sCache  (sKernel* sk, int32_t Mbyte, int32_t ell);
  ~sCache ();

  cachetype *FillRow (int32_t row, int32_t IsC = 0);
  cachetype *GetRow  (int32_t row);

  int32_t DivideMP (int32_t *out, int32_t *in, int32_t n);

  /*** Itarations counter ***/
  void Iteration() { nit++; }

  /*** Cache size control ***/
  int32_t CheckCycle()
  {
    int32_t us;
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
    int32_t row;      // unused row
    int32_t last_access_it;
    cache_entry *prev, *next;
    cachetype   *data;
  };

  sKernel* KER;
  int32_t maxmw, ell;
  int32_t nit;

  cache_entry *mw;
  cache_entry *first_free;
  cache_entry **pindmw;    // 0 if unused row
  cachetype   *onerow;

  cachetype   *FindFree(int32_t row, int32_t IsC);
};


/******************************************************************************/
/*** Cache class constructor                                                ***/
/******************************************************************************/
sCache::sCache(sKernel* sk, int32_t Mbyte, int32_t _ell) : KER(sk), ell(_ell)
{
  int32_t i;

  // size in dwords of one cache row
  maxmw = (sizeof(cache_entry) + sizeof(cache_entry *)
           + ell*sizeof(cachetype)) / 4;
  // number of cache rows
  maxmw = Mbyte*1024*(1024/4) / maxmw;

  /* arrays allocation */
  mw     = SG_MALLOC(cache_entry, maxmw);
  pindmw = SG_MALLOC(cache_entry*,  ell);
  onerow = SG_MALLOC(cachetype,     ell);

  /* arrays initialization */
  for (i = 0; i < maxmw; i++)
  {
      mw[i].prev           = (i == 0 ? &mw[maxmw-1] : &mw[i-1]);
      mw[i].next           = (i == maxmw-1 ? &mw[0] : &mw[i+1]);
      mw[i].data           = SG_MALLOC(cachetype, ell);
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
  int32_t i;

  for (i = maxmw-1; i >= 0; i--)
      SG_FREE(mw[i].data);

  SG_FREE(onerow);
  SG_FREE(pindmw);
  SG_FREE(mw);
}


/******************************************************************************/
/*** Retrieve a cached row                                                  ***/
/******************************************************************************/
cachetype *sCache::GetRow(int32_t row)
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
cachetype *sCache::FindFree(int32_t row, int32_t IsC)
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
cachetype *sCache::FillRow(int32_t row, int32_t IsC)
{
  int32_t j;
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
int32_t sCache::DivideMP(int32_t *out, int32_t *in, int32_t n)
{
   /********************************************************************
    * Input meaning:                                                   *
    *    in  = vector containing row to be extracted in the cache      *
    *    n   = size of in                                              *
    *    out = the indexes of "in" of the components to be computed    *
    *          by this processor (first those in the cache, then the   *
    *          ones not yet computed)                                  *
    * Returns: the number of components of this processor              *
    ********************************************************************/

  int32_t *remained, nremained, k;
  int32_t i;

  remained = SG_MALLOC(int32_t, n);

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

  SG_FREE(remained);
  return n;
}

/******************************************************************************/
/*** Check solution optimality                                              ***/
/******************************************************************************/
int32_t QPproblem::optimal()
{
  /***********************************************************************
   * Returns 1 if the computed solution is optimal, otherwise returns 0. *
   * To verify the optimality it checks the KKT optimality conditions.   *
   ***********************************************************************/
  register int32_t i, j, margin_sv_number, z, k, s, kin, z1, znew=0, nnew;

  float64_t gx_i, aux, s1, s2;

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
          SG_SINFO("  Max KKT violation: %lf\n", aux)
      else if (verbosity > 0)
          SG_SINFO("  %lf\n", aux)

      if (fabs(kktold-aux) < delta*0.01 &&  aux < delta*2.0)
      {
          if (DELTAvpm > InitialDELTAvpm*0.1)
          {
              DELTAvpm = (DELTAvpm*0.5 > InitialDELTAvpm*0.1 ?
                                            DELTAvpm*0.5 : InitialDELTAvpm*0.1);
              SG_SINFO("Inner tolerance changed to: %lf\n", DELTAvpm)
          }
      }

      kktold = aux;

 /*****************************************************************************
  *** Update the working set (T. Serafini, L. Zanni, "On the Working Set    ***
  *** Selection in Gradient Projection-based Decomposition Techniques for   ***
  *** Support Vector Machines"; Optim. Meth. Soft. 20, 2005).               ***
  *****************************************************************************/
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
          if (i == z1) break;

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
          if (znew == s) break;

          index_in[k++] = ing[z];
          index_in[k++] = ing[z1];
      }

      if (k < q)
      {
          if (verbosity > 1)
              SG_SINFO("  New q: %i\n", k)
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

      if (kin == 0)
      {
          DELTAkin *= 0.1;
          if (DELTAkin < 1.0e-6)
          {
              SG_SINFO("\n***ERROR***: GPDT stops with tolerance")
              SG_SINFO(
              " %lf because it is unable to change the working set.\n", kktold);
              return 1;
          }
          else
          {
              SG_SINFO("Inner tolerance temporary changed to:")
              SG_SINFO(" %e\n", DELTAvpm*DELTAkin)
          }
      }
      else
          DELTAkin = 1.0;

      if (verbosity > 1)
      {
          SG_SINFO("  Working set: new components: %i", kin)
          SG_SINFO(",  new parameter n: %i\n", q)
      }

      return 0;
   }
}

/******************************************************************************/
/*** Optional preprocessing: random distribution                            ***/
/******************************************************************************/
int32_t QPproblem::Preprocess0(int32_t *aux, int32_t *sv)
{
  int32_t i, j;

  Randnext = 1;
  memset(sv, 0, ell*sizeof(int32_t));
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
int32_t QPproblem::Preprocess1(sKernel* kernel, int32_t *aux, int32_t *sv)
{
  int32_t    s;    // elements owned by the processor
  int32_t    sl;   // elements of the n-1 subproblems
  int32_t    n, i, off, j, k, ll;
  int32_t    nsv, nbsv;
  int32_t    *sv_loc, *bsv_loc, *sp_y;
  float32_t  *sp_D=NULL;
  float64_t *sp_alpha, *sp_h;

  s  = ell;
  /* divide the s elements into n blocks smaller than preprocess_size */
  n  = (s + preprocess_size - 1) / preprocess_size;
  sl = 1 + s / n;

  if (verbosity > 0)
  {
      SG_SINFO("  Preprocessing: examples = %d", s)
      SG_SINFO(", subp. = %d", n)
      SG_SINFO(", size = %d\n",sl)
  }

  sv_loc   = SG_MALLOC(int32_t, s);
  bsv_loc  = SG_MALLOC(int32_t, s);
  sp_alpha = SG_MALLOC(float64_t, sl);
  sp_h     = SG_MALLOC(float64_t, sl);
  sp_y     = SG_MALLOC(int32_t, sl);

  if (sl < 500)
      sp_D = SG_MALLOC(float32_t, sl*sl);

  for (i = 0; i < sl; i++)
       sp_h[i] = -1.0;
  memset(alpha, 0, ell*sizeof(float64_t));

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
          SG_SINFO("%d...", i)
      SplitParts(s, i, n, &ll, &off);

      if (sl < 500)
      {
          for (j = 0; j < ll; j++)
          {
              sp_y[j] = y[aux[j+off]];
              for (k = j; k < ll; k++)
                  sp_D[k*sl + j] = sp_D[j*sl + k]
                                 = y[aux[j+off]] * y[aux[k+off]]
                                   * (float32_t)kernel->Get(aux[j+off], aux[k+off]);
          }

          memset(sp_alpha, 0, sl*sizeof(float64_t));

          /* call the gradient projection QP solver */
          gpm_solver(projection_solver, projection_projector, ll, sp_D, sp_h,
                     c_const, 0.0, sp_y, sp_alpha, delta*10, NULL);
      }
      else
      {
          QPproblem p2;
		  QPproblem::copy_subproblem(&p2, this, ll, aux + off);
          p2.chunk_size     = (int32_t) ((float64_t)chunk_size / sqrt((float64_t)n));
          p2.q              = (int32_t) ((float64_t)q / sqrt((float64_t)n));
          p2.maxmw          = ll*ll*4 / (1024 * 1024);
          if (p2.maxmw > maxmw / 2)
              p2.maxmw = maxmw / 2;
          p2.verbosity      = 0;
          p2.delta          = delta * 10.0;
          p2.PreprocessMode = 0;
          kernel->KernelEvaluations += p2.gpdtsolve(sp_alpha);
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
  memset(sv, 0, ell*sizeof(int32_t));
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

  /* eventually fill up the working set with other components
     randomly chosen                                          */
  for (; i < chunk_size; i++)
  {
      do {
           j = ThRandPos % ell;
      } while (sv[j] != 0);
      sv[j] = 1;
  }


  /* dealloc temporary arrays */
  if (sl < 500) SG_FREE(sp_D);
  SG_FREE(sp_y    );
  SG_FREE(sp_h    );
  SG_FREE(sv_loc  );
  SG_FREE(bsv_loc );
  SG_FREE(sp_alpha);

  if (verbosity > 0)
  {
      SG_SINFO("\n  Preprocessing: SV = %d", nsv)
      SG_SINFO(", BSV = %d\n", nbsv)
  }

  return(nsv);
}

/******************************************************************************/
/*** Compute the QP problem solution                                        ***/
/******************************************************************************/
float64_t QPproblem::gpdtsolve(float64_t *solution)
{
  int32_t i, j, k, z, iin, jin, nit, tot_vpm_iter, lsCount;
  int32_t tot_vpm_secant, projCount, proximal_count;
  int32_t vpmWarningThreshold;
  int32_t  nzin, nzout;
  int32_t *sp_y;               /* labels vector                             */
  int32_t *indnzin, *indnzout; /* nonzero components indices vectors        */
  float32_t     *sp_D;               /* quadratic part of the objective function  */
  float64_t    *sp_h, *sp_hloc,     /* linear part of the objective function     */
            *sp_alpha,*stloc;    /* variables and gradient updating vectors   */
  float64_t    sp_e, aux, fval, tau_proximal_this, dfval;
  float64_t    *vau;
  float64_t    *weight;
  float64_t    tot_prep_time, tot_vpm_time, tot_st_time, total_time;
  sCache    *Cache;
  cachetype *ptmw;
  clock_t   t, ti;

  Cache = new sCache(KER, maxmw, ell);
    if (chunk_size > ell) chunk_size = ell;

  if (chunk_size <= 20)
      vpmWarningThreshold = 30*chunk_size;
  else if (chunk_size <= 200)
      vpmWarningThreshold = 20*chunk_size + 200;
  else
      vpmWarningThreshold = 10*chunk_size + 2200;

  kktold = 10000.0;
  if (delta <= 5e-3)
  {
      if ( (chunk_size <= 20) | ((float64_t)chunk_size/ell <= 0.001) )
          DELTAvpm = delta * 0.1;
      else if ( (chunk_size <= 200) | ((float64_t)chunk_size/ell <= 0.005) )
          DELTAvpm = delta * 0.5;
      else
          DELTAvpm = delta;
  }
  else
  {
      if ( (chunk_size <= 200) | ((float64_t)chunk_size/ell <= 0.005) )
          DELTAvpm = (1e-3 < delta*0.1) ? 1e-3 : delta*0.1;
      else
          DELTAvpm = 5e-3;
  }

  InitialDELTAvpm = DELTAvpm;
  DELTAsv         = EPS_SV * c_const;
  DELTAkin        = 1.0;

  q               = q & (~1);
  nb              = ell - chunk_size;
  tot_vpm_iter    = 0;
  tot_vpm_secant  = 0;

  tot_prep_time = tot_vpm_time = tot_st_time = total_time = 0.0;

  ti = clock();

  /* arrays allocation */
  SG_SDEBUG("ell:%d, chunk_size:%d, nb:%d dim:%d\n", ell, chunk_size,nb, dim)
  ing       = SG_MALLOC(int32_t, ell);
  inaux     = SG_MALLOC(int32_t, ell);
  index_in  = SG_MALLOC(int32_t, chunk_size);
  index_out = SG_MALLOC(int32_t, ell);
  indnzout  = SG_MALLOC(int32_t, nb);
  alpha     = SG_MALLOC(float64_t, ell);

  memset(alpha, 0, ell*sizeof(float64_t));
  memset(ing,   0, ell*sizeof(int32_t));

  if (verbosity > 0 && PreprocessMode != 0)
      SG_SINFO("\n*********** Begin setup step...\n")
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
      SG_SINFO("  Time for setup: %.2lf\n", (float64_t)t/CLOCKS_PER_SEC)
      SG_SINFO(
              "\n\n*********** Begin decomposition technique...\n");
  }

  /* arrays allocation */
  bmem     = SG_MALLOC(int32_t, ell);
  bmemrid  = SG_MALLOC(int32_t, chunk_size);
  pbmr     = SG_MALLOC(int32_t, chunk_size);
  cec      = SG_MALLOC(int32_t, ell);
  indnzin  = SG_MALLOC(int32_t, chunk_size);
  inold    = SG_MALLOC(int32_t, chunk_size);
  incom    = SG_MALLOC(int32_t, chunk_size);
  vau      = SG_MALLOC(float64_t, ell);
  grad     = SG_MALLOC(float64_t, ell);
  weight   = SG_MALLOC(float64_t, dim);
  st       = SG_MALLOC(float64_t, ell);
  stloc    = SG_MALLOC(float64_t, ell);

  for (i = 0; i < ell; i++)
  {
      bmem[i] = 0;
      cec[i]  = 0;
      st[i]   = 0;
  }

  sp_y     = SG_MALLOC(int32_t, chunk_size);
  sp_D     = SG_MALLOC(float32_t, chunk_size*chunk_size);
  sp_alpha = SG_MALLOC(float64_t, chunk_size);
  sp_h     = SG_MALLOC(float64_t, chunk_size);
  sp_hloc  = SG_MALLOC(float64_t, chunk_size);

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
      SG_SINFO("  IT  | Prep Time | Solver IT | Solver Time |")
      SG_SINFO(" Grad Time | KKT violation\n")
      SG_SINFO("------+-----------+-----------+-------------+")
      SG_SINFO("-----------+--------------\n")
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
              total_time += (float64_t)(t-ti) / CLOCKS_PER_SEC;
          else
              total_time += (float64_t)(ti-t) / CLOCKS_PER_SEC;
          ti = t;
      }

      if (verbosity > 1)
          SG_SINFO("\n*********** ITERATION: %d\n", nit + 1)
      else if (verbosity > 0)
          SG_SINFO("%5d |", nit + 1)
      else
          SG_SINFO(".")
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
          SG_SINFO("  spe: %e ", sp_e)

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
                                           * (float32_t)KER->Get(iin, index_in[j]);
          else
          {
              for (j = 0; j < i; j++)
                  if (incom[j] == -1)
                      sp_D[i*chunk_size + j]
                         = sp_y[i]*sp_y[j] * (float32_t)KER->Get(iin, index_in[j]);
                  else
                      sp_D[i*chunk_size + j]
                         = sp_D[incom[j]*chunk_size + incom[i]];
              sp_D[i*chunk_size + i]
                  = sp_y[i]*sp_y[i] * (float32_t)KER->Get(iin, index_in[i]);
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

    t = clock() - t;
    if (verbosity > 1)
        SG_SINFO("  Preparation Time: %.2lf\n", (float64_t)t/CLOCKS_PER_SEC)
    else if (verbosity > 0)
        SG_SINFO("  %8.2lf |", (float64_t)t/CLOCKS_PER_SEC)
    tot_prep_time += (float64_t)t/CLOCKS_PER_SEC;

    /*** Proximal point modification: first type ***/

    if (tau_proximal < 0.0)
      tau_proximal_this = 0.0;
    else
      tau_proximal_this = tau_proximal;
    proximal_count = 0;
    do {
      t = clock();
      for (i = 0; i < chunk_size; i++)
      {
          vau[i]                  = sp_D[i*chunk_size + i];
          sp_h[i]                -= tau_proximal_this * alpha_in(i);
          sp_D[i*chunk_size + i] += (float32_t)tau_proximal_this;
      }

      if (kktold < delta*100)
          for (i = 0; i < chunk_size; i++)
              sp_alpha[i] = alpha_in(i);
      else
          for (i = 0; i < chunk_size; i++)
              sp_alpha[i] = 0.0;

      /*** call the chosen inner gradient projection QP solver ***/
      i = gpm_solver(projection_solver, projection_projector, chunk_size,
                    sp_D, sp_h, c_const, sp_e, sp_y, sp_alpha,
                    DELTAvpm*DELTAkin, &lsCount, &projCount);

      if (i > vpmWarningThreshold)
      {
        if (ker_type == 2)
        {
            SG_SINFO("\n WARNING: inner subproblem hard to solve;")
            SG_SINFO(" setting a smaller -q or")
            SG_SINFO(" tuning -c and -g options might help.\n")
        }
        else
        {
            SG_SINFO("\n WARNING: inner subproblem hard to solve;")
            SG_SINFO(" set a smaller -q or")
            SG_SINFO(" try a better data scaling.\n")
        }
      }

      t = clock() - t;
      tot_vpm_iter   += i;
      tot_vpm_secant += projCount;
      tot_vpm_time   += (float64_t)t/CLOCKS_PER_SEC;
      if (verbosity > 1)
      {
          SG_SINFO("  Solver it: %d", i)
          SG_SINFO(", ls: %d", lsCount)
          SG_SINFO(", time: %.2lf\n", (float64_t)t/CLOCKS_PER_SEC)
      }
      else if (verbosity > 0)
      {
          SG_SINFO("    %6d", i)
          SG_SINFO(" |    %8.2lf |", (float64_t)t/CLOCKS_PER_SEC)
      }

      /*** Proximal point modification: second type ***/

      for (i = 0; i < chunk_size; i++)
          sp_D[i*chunk_size + i] = (float32_t)vau[i];
      tau_proximal_this = 0.0;
      if (tau_proximal < 0.0)
      {
        dfval = 0.0;
        for (i = 0; i < chunk_size; i++)
        {
          aux = 0.0;
          for (j = 0; j < chunk_size; j++)
            aux += sp_D[i*chunk_size + j]*(alpha_in(j) - sp_alpha[j]);
          dfval += (0.5*aux - st[index_in[i]]*y_in(i) + 1.0) * (alpha_in(i) - sp_alpha[i]);
        }

        aux=0.0;
        for (i = 0; i < chunk_size; i++)
            aux +=  (alpha_in(i) - sp_alpha[i])*(alpha_in(i) - sp_alpha[i]);

        if ((-dfval/aux) < -0.5*tau_proximal)
        {
          tau_proximal_this = -tau_proximal;
          if (verbosity > 0)
            SG_SDEBUG("tau threshold: %lf  ", -dfval/aux)
        }
      }
      proximal_count++;
    } while (tau_proximal_this != 0.0 && proximal_count < 2); // Proximal point loop

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

	// in case of LINADD enabled use faster linadd variant
	if (KER->get_kernel()->has_property(KP_LINADD) && get_linadd_enabled())
	{
		KER->get_kernel()->clear_normal() ;

		for (j = 0; j < nzin; j++)
			KER->get_kernel()->add_to_normal(indnzin[j], grad[j]);

        if (nit == 0 && PreprocessMode > 0)
		{
			for (j = 0; j < nzout; j++)
			{
				jin = indnzout[j];
				KER->get_kernel()->add_to_normal(jin, alpha[jin] * y[jin]);
			}
		}

        for (i = 0; i < ell; i++)
            st[i] += KER->get_kernel()->compute_optimized(i);
	}
	else  // nonlinear kernel
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
                SG_SINFO(
                 "  G*x0 time: %.2lf\n", (float64_t)(clock()-ti2)/CLOCKS_PER_SEC);
        }
    }

    /*** sort the vectors for cache managing ***/

    t = clock() - t;
    if (verbosity > 1)
        SG_SINFO("  Gradient updating time: %.2lf\n", (float64_t)t/CLOCKS_PER_SEC)
    else if (verbosity > 0)
        SG_SINFO("  %8.2lf |", (float64_t)t/CLOCKS_PER_SEC)
    tot_st_time += (float64_t)t/CLOCKS_PER_SEC;

    /*** global updating of the solution vector ***/
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
        SG_SINFO("  SV: %d", j+k)
        SG_SINFO(",  BSV: %d\n", k)
    }
    Cache->Iteration();
    nit = nit+1;
  } while (!optimal() && !(CSignal::cancel_computations()));
  /* End of the problem resolution loop                                      */
  /***************************************************************************/

  t = clock();
  if ((t-ti) > 0)
      total_time += (float64_t)(t-ti) / CLOCKS_PER_SEC;
  else
      total_time += (float64_t)(ti-t) / CLOCKS_PER_SEC;
  ti = t;

  memcpy(solution, alpha, ell * sizeof(float64_t));

  /* objective function evaluation */
  fval = 0.0;
  for (i = 0; i < ell; i++)
      fval += alpha[i]*(y[i]*st[i]*0.5 - 1.0);

  SG_SINFO("\n------+-----------+-----------+-------------+")
  SG_SINFO("-----------+--------------\n")
  SG_SINFO(
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
      SG_SINFO(
        "- Variables entering the working set at least one time:  %i\n", j);
      SG_SINFO(
        "- Variables entering the working set at least two times:  %i\n", k);
      SG_SINFO(
        "- Variables entering the working set at least three times:  %i\n", z);
  }


  SG_FREE(bmem);
  SG_FREE(bmemrid);
  SG_FREE(pbmr);
  SG_FREE(cec);
  SG_FREE(ing);
  SG_FREE(inaux);
  SG_FREE(indnzin);
  SG_FREE(index_in);
  SG_FREE(inold);
  SG_FREE(incom);
  SG_FREE(indnzout);
  SG_FREE(index_out);
  SG_FREE(vau);
  SG_FREE(alpha);
  SG_FREE(weight);
  SG_FREE(grad);
  SG_FREE(stloc);
  SG_FREE(st);
  SG_FREE(sp_h);
  SG_FREE(sp_hloc);
  SG_FREE(sp_y);
  SG_FREE(sp_D);
  SG_FREE(sp_alpha);
  delete Cache;

  aux = KER->KernelEvaluations;
  SG_SINFO("- Total CPU time: %lf\n", total_time)
  if (verbosity > 0)
  {
      SG_SINFO(
              "- Total kernel evaluations: %.0lf\n", aux);
      SG_SINFO(
              "- Total inner solver iterations: %i\n", tot_vpm_iter);
      if (projection_projector == 1)
          SG_SINFO(
              "- Total projector iterations: %i\n", tot_vpm_secant);
      SG_SINFO(
              "- Total preparation time: %lf\n", tot_prep_time);
      SG_SINFO(
              "- Total inner solver time: %lf\n", tot_vpm_time);
      SG_SINFO(
              "- Total gradient updating time: %lf\n", tot_st_time);
  }
  SG_SINFO("- Objective function value: %lf\n", fval)
  objective_value=fval;
  return aux;
}

/******************************************************************************/
/*** Quick sort for integer vectors                                         ***/
/******************************************************************************/
void quick_si(int32_t a[], int32_t n)
{
  int32_t i, j, s, d, l, x, w, ps[20], pd[20];

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
void quick_s2(float64_t a[], int32_t n, int32_t ia[])
{
  int32_t     i, j, s, d, l, iw, ps[20], pd[20];
  float64_t  x, w;

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
void quick_s3(int32_t a[], int32_t n, int32_t ia[])
{
  int32_t i, j, s, d, l, iw, w, x, ps[20], pd[20];

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
}

#endif // DOXYGEN_SHOULD_SKIP_THIS

/******************************************************************************/
/*** End of gpdtsolve.cpp file                                              ***/
/******************************************************************************/
