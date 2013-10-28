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
 *** File:     gpdt.cpp                                                     ***
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
#include <ctype.h>
#include <math.h>
#include <shogun/lib/external/gpdt.h>
#include <shogun/lib/external/gpdtsolve.h>

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
void    fatalError(const char *msg1, const char *msg2);

/******************************************************************************/
/*** Class constructor                                                      ***/
/******************************************************************************/
QPproblem::QPproblem()
{
  /*** set problem defaults ***/
  maxmw                = 40;
  c_const              = 10.0;
  projection_solver    = SOLVER_FLETCHER;
  projection_projector = 1;
  PreprocessMode       = 0;
  delta                = 1.0e-3;
  DELTAsv              = EPS_SV;
  ker_type             = 2;
  chunk_size           = 400;
  q                    = -1;
  y                    = NULL;
  tau_proximal         = 0.0;
  dim = 1;
}

/******************************************************************************/
/*** Class destructor                                                       ***/
/******************************************************************************/
QPproblem::~QPproblem()
{
  //if (y != NULL) free(y);
}

/******************************************************************************/
/*** Setter method for the subproblem features                              ***/
/******************************************************************************/
void QPproblem::copy_subproblem(QPproblem* dst, QPproblem* p, int32_t len, int32_t *perm)
{
  int32_t k;

  *dst=*p;
  dst->ell = len;

  dst->KER->SetSubproblem(p->KER, len, perm);
  dst->y = SG_MALLOC(int32_t, len);
  for (k = 0; k < len; k++)
      dst->y[k] = p->y[perm[k]];
}

namespace shogun
{
/******************************************************************************/
/*** Extract the samples information from an SVMlight-compliant data file   ***/
/******************************************************************************/
int32_t prescan_document(char *file, int32_t *lines, int32_t *vlen, int32_t *ll)
{
  FILE    *fl;
  int32_t ic;
  char    c;
  int64_t    current_length, current_vlen;

  if ((fl = fopen (file, "r")) == NULL)
      return(-1);
  current_length = 0;
  current_vlen   = 0;

  *ll    = 0;  /* length of the longest input line (the read buffer should
                  be allocated with this size)                              */
  *lines = 1;  /* number of lines in the file                               */
  *vlen  = 0;  /* max number of nonzero components in a vector              */

  while ((ic = getc(fl)) != EOF)
  {
    c = (char)ic;
    current_length++;

    if (c == ' ')
        current_vlen++;

    if (c == '\n')
    {
        (*lines)++;
        if (current_length > (*ll))
            *ll = current_length;
        if (current_vlen > (*vlen))
            *vlen = current_vlen;
        current_length = 0;
        current_vlen   = 0;
    }
  }
  fclose(fl);
  return(0);
}
}
/******************************************************************************/
/*** return 1 if problem is single class, 0 if two-class                    ***/
/******************************************************************************/
int32_t QPproblem::Check2Class()
{
  int32_t i;

  for (i = 1; i < ell; i++)
      if (y[i] != y[0])
          return 0;
  return 1;
}

namespace shogun
{
/******************************************************************************/
/*** Compute the size of data splitting for preprocessing                   ***/
/******************************************************************************/
void SplitParts(
	int32_t n, int32_t part, int32_t parts, int32_t *dim, int32_t *off)
{
  int32_t r;

  r    = n % parts;
  *dim = n / parts;

  if (part < r)
  {
     (*dim)++;
     *off = *dim * part;
  }
  else
     *off = *dim * part + r;
}
}
/******************************************************************************/
/*** Kernel class constructor                                               ***/
/******************************************************************************/
sKernel::sKernel (CKernel* k, int32_t l)
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
void sKernel::SetData(
	float32_t **x_, int32_t **ix_, int32_t *lx_, int32_t _ell, int32_t _dim)
{
  int32_t i, j, k;

  dim  = _dim;
  ell  = _ell;
  nor  = SG_MALLOC(float64_t, ell);
  vaux = SG_CALLOC(float32_t, dim);

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
          nor[i] += (float64_t)(x[i][j]*x[i][j]);
  }
}

/******************************************************************************/
/*** Set the subproblem data                                                ***/
/******************************************************************************/
void sKernel::SetSubproblem(sKernel* ker, int32_t len, int32_t *perm)
{
  int32_t k;

  /* arrays allocations */
  nor  = SG_MALLOC(float64_t, len);
  vaux = SG_CALLOC(float32_t, ker->dim);

  lx = SG_MALLOC(int32_t, len);
  ix = SG_MALLOC(int32_t*, len);
  x  = SG_MALLOC(float32_t*, len);
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
/*** Kernel class destructor                                                ***/
/******************************************************************************/
sKernel::~sKernel()
{
  int32_t i;

  SG_FREE(nor);
  SG_FREE(vaux);

  SG_FREE(lx);
  if (ix != NULL)
  {
      if (!IsSubproblem)
          for (i = 0; i < ell; i++)
              SG_FREE(ix[i]);
      SG_FREE(ix);
  }
  if (x != NULL)
  {
      if (!IsSubproblem)
          for (i = 0; i < ell; i++)
              SG_FREE(x[i]);
      SG_FREE(x);
  }
}

#endif // DOXYGEN_SHOULD_SKIP_THIS

/******************************************************************************/
/*** End of gpdt.cpp file                                                   ***/
/******************************************************************************/
