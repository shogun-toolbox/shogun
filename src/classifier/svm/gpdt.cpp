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
 *** File:     gpdt.cpp                                                     ***
 *** Type:     scalar                                                       ***
 *** Version:  0.9 beta                                                     ***
 *** Date:     July 21, 2004                                                ***
 *** Revision: 1                                                            ***
 ***                                                                        ***
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "gpdt.h"

char    cOutputStream[10000][80]; /* buffer array to store output messages */
int     nOutputStream;

#define OutputStream (cOutputStream[nOutputStream++])

void    help_message(void);
void    fatalError(const char *msg1, const char *msg2);

/******************************************************************************/
/*** Main method                                                            ***/
/******************************************************************************/
//see GPBTSVM.cpp
/******************************************************************************/
/*** Display the program invocation syntax                                  ***/
/******************************************************************************/
void help_message(void)
{
  fprintf(stderr, "usage: gpdt [options] example_file model_file\n");
  fprintf(stderr, "option:\n");
  fprintf(stderr, "   -? this help\n");
  fprintf(stderr, "   -h display help message\n");
  fprintf(stderr, "   -v [0..2] verbosity level (default 1)\n");
  fprintf(stderr, "   -t [0..2] type of kernel function (default 2):\n");
  fprintf(stderr, "       0: linear (xT y)\n");
  fprintf(stderr, "       1: polynomial (s(xT y) + r)d\n");
  fprintf(stderr, "       2: radial basis function (rbf): exp(-g||x - y||2)\n");
  fprintf(stderr, "   -s parameter s in polynomial kernel (default 1.0)\n");
  fprintf(stderr, "   -r parameter r in polynomial kernel (default 1.0)\n");
  fprintf(stderr, "   -d parameter d in polynomial kernel (default 3)\n");
  fprintf(stderr, "   -g parameter g in rbf kernel (default 1.0)\n");
  fprintf(stderr, "   -c parameter C for SVM classification: trade-off between\n");
  fprintf(stderr, "      training error and margin (default 1.0)\n");
  fprintf(stderr, "   -q size of the QP-subproblems: q . 2 (default 400)\n");
  fprintf(stderr, "   -n maximum number of new indices entering the working set\n");
  fprintf(stderr, "      in each iteration: 2 <= n <= q, n even (default q/3)\n");
  fprintf(stderr, "   -e tolerance for termination criterion (default 0.001)\n");
  fprintf(stderr, "   -a [0, 1] gradient projection-type inner QP solver:\n");
  fprintf(stderr, "      0: Generalized Variable Projection method\n");
  fprintf(stderr, "      1: Dai-Fletcher Projected Gradient method (default)\n");
  fprintf(stderr, "   -m cache size in MB (default 40)\n");
  fprintf(stderr, "   -u parameter for proximal point modification (default 0)\n");
  exit(-1);
}

/******************************************************************************/
/*** Class constructor                                                      ***/
/******************************************************************************/
QPproblem::QPproblem()
{
  /*** set problem defaults ***/
  maxmw             = 40;
  c_const           = 10.0;
  projection_solver = SOLVER_FLETCHER;
  PreprocessMode    = 0;
  delta             = 1.0e-3;
  DELTAsv           = EPS_SV;
  ker_type          = 2;
  chunk_size        = 400;
  q                 = -1;
  y                 = NULL;
  tau_proximal      = 0.0;
}

/******************************************************************************/
/*** Class destructor                                                       ***/
/******************************************************************************/
QPproblem::~QPproblem()
{
  if (y != NULL) free(y);
}

/*** setter method for the subproblem features ***/
void QPproblem::Subproblem(QPproblem &p, int len, int *perm)
{
  int k;

  memcpy(this, &p, sizeof(QPproblem));
  ell = len;

  KER->SetSubproblem(p.KER, len, perm);
  y = (int *)malloc(len * sizeof(int));
  for (k = 0; k < ell; k++)
      y[k] = p.y[perm[k]];
}


/******************************************************************************/
/*** Extract the samples information from an SVMlight-compliant data file   ***/
/******************************************************************************/
int prescan_document(char *file, int *lines, int *vlen, int *ll)
{
  FILE  *fl;
  int   ic;
  char  c;
  long  current_length, current_vlen;

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

    if(c == ' ')
      current_vlen++;

    if(c == '\n')
    {
      (*lines)++;
      if (current_length > (*ll))
        *ll = current_length;
      if (current_vlen > (*vlen))
        *vlen = current_vlen;
      current_length=0;
      current_vlen=0;
    }
  }
  fclose(fl);
  return(0);
}


/******************************************************************************/
/*** Read the training data from an SVMlight-compliant file                 ***/
/******************************************************************************/
int QPproblem::ReadSVMFile(char *f1_input)
{
  char   *line;
  int    i, j, ell_space, vlen, max_row_length;
  int    *line_ix;
  int    **data_ix, *data_lx;
  float  *line_x;
  float  **data_x;
  FILE   *fp1_in;

  /*** read the data global information ***/
  if (prescan_document(f1_input, &ell_space, &vlen, &max_row_length))
      return(-1);

  ell_space      += 10;
  vlen           += 10;
  max_row_length += 10;

  /*** arrays allocation ***/
  dim     = 0;
  ell     = 0;
  data_lx = (int    *)malloc(ell_space*sizeof(int      ));
  data_ix = (int   **)malloc(ell_space*sizeof(int *    ));
  data_x  = (float **)malloc(ell_space*sizeof(float *  ));
  y       = (int    *)malloc(ell_space*sizeof(int      ));
  line    = (char   *)malloc(max_row_length*sizeof(char));
  line_ix = (int    *)malloc(vlen * sizeof(int         ));
  line_x  = (float  *)malloc(vlen * sizeof(float       ));

  /*** open the training data file for input ***/
  fp1_in = fopen(f1_input, "r");
  if (fp1_in == NULL)
      return(-1);

  /*** start reading the training data ***/
  fgets(line, max_row_length, fp1_in);

  while (!feof(fp1_in))
  {
    for (i = 0; line[i] != 0 && line[i] != '#'; i++);
    line[i] = 0;  // remove comments

    if (sscanf(line, "%d", &j) != EOF)   // read the sample label
    {
      y[ell] = j;

      j = i = 0;
      while (line[i] == ' ' || line[i] == '\t') i++;
      while (line[i] > ' ') i++;
      while (sscanf(line+i, "%d:%f", &line_ix[j], &line_x[j]) != EOF)
      {
        while (line[i] == ' ' || line[i] == '\t') i++;
        while (line[++i] > ' ');
        j++;
      }

      data_lx[ell] = j;
      if (data_lx[ell] > 0)  // read in a nontrivial sample
      {
          data_ix[ell] = (int   *)malloc(data_lx[ell]*sizeof(int  ));
          data_x[ell]  = (float *)malloc(data_lx[ell]*sizeof(float));

          memcpy(data_ix[ell], line_ix, data_lx[ell]*sizeof(int  ));
          memcpy(data_x[ell],  line_x,  data_lx[ell]*sizeof(float));

          if (dim < (data_ix[ell][data_lx[ell]-1] + 1))
              dim = data_ix[ell][data_lx[ell]-1] + 1;
      }
      else
      {
          data_ix[ell] = (int   *)malloc(sizeof(int  ));
          data_x[ell]  = (float *)malloc(sizeof(float));
          *(data_ix[ell]) = 0;
          *(data_x[ell])  = 0.0;
      }
      ell++;

      if (verbosity > 1)
          if ((ell % 1000) == 0)
              fprintf(stderr, " %d...", ell);
    }
    fgets(line, max_row_length, fp1_in);
  }
  fclose(fp1_in); // training data end

  /*** check for dimensions consistency ***/
  if (chunk_size > ell)
      chunk_size = ell;
  if (q > chunk_size)
      q = chunk_size;

  /*** buffers freeing ***/
  free(line_x);
  free(line_ix);
  free(line);

  /*** set the data in the kernel object ***/
  KER->SetData(data_x, data_ix, data_lx, ell, dim);

  return(0);
}

/******************************************************************************/
/*** Compute the size of data splitting for preprocessing                   ***/
/******************************************************************************/
void SplitParts(int n, int part, int parts, int *dim, int *off)
{
  int  r;

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

/******************************************************************************/
/*** Utility method for fatal error messages and halt                       ***/
/******************************************************************************/
void fatalError(const char* msg1, const char* msg2)
{
  fprintf(stderr, ">>> FATAL ERROR: %s\n\t\t%s\n", msg1, msg2);
  exit(-1);
}

/******************************************************************************/
/*** End of gpdt.cpp file                                                   ***/
/******************************************************************************/
