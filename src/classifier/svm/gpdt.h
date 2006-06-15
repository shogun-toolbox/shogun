/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

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
 *** File:     gpdt.h                                                       ***
 *** Type:     scalar                                                       ***
 *** Version:  1.0                                                          ***
 *** Date:     October, 2005                                                ***
 *** Revision: 1                                                            ***
 ***                                                                        ***
 ******************************************************************************/
#include "kernel/Kernel.h"

#define MAXLENGTH 256
#define cachetype KERNELCACHE_ELEM
#define EPS_SV    1.0e-9   /* precision for multipliers */

enum {
  SOLVER_VPM      = 0,
  SOLVER_FLETCHER = 1
};

void output_message(const char *msg);
void output_message(const char *msg, int v);
void output_message(const char *msg, double v);
int gpm_solver(int Solver, int Projector, int n, float *A, double *b, double c,
               double e, int *iy, double *x, double tol, 
               int *ls = 0, int *proj = 0);

class sKernel
{
public:
  int    ker_type;
  int    *lx;
  int    **ix;
  float  **x;
  double *nor;
  double sigma;
  double degree;
  double norm;    // normalization factor
  double c_poly;
  double KernelEvaluations;

  double (sKernel::*kernel_fun)(int i, int j);

  sKernel (CKernel* k, int ell);
  ~sKernel();

  void   SetData       (float **x_, int **ix_, int *lx_, int ell, int dim);
  void   SetSubproblem (sKernel* ker, int len, int *perm);
  void   SetKernel     (int type, double sigma_, double degree_, 
                        double normalisation, double cp);
  double Get(int i, int j)
  {
    KernelEvaluations += 1.0F;
    return kernel->kernel(i, j);
  }
  void   Add           (double *v, int j, double mul);
  double Prod          (double *v, int j);

private:
  CKernel* kernel;
  int    vauxRow;
  int    IsSubproblem;
  int    ell, dim;
  float  *vaux;

  double dot     (int i, int j);

  double k_lin   (int i, int j);
  double k_gauss (int i, int j);
  double k_pow   (int i, int j);
  double k_pow2  (int i, int j);
  double k_pow3  (int i, int j);
  double k_pow4  (int i, int j);
  double k_pow5  (int i, int j);
  double k_pow6  (int i, int j);
  double k_pow7  (int i, int j);
  double k_pow8  (int i, int j);
  double k_pow9  (int i, int j);
};


class QPproblem
{
// ----------------- Public Data ---------------
public:
  int     chunk_size;
  int     ell;
  int    *y;
  double DELTAsv;
  int     q;
  int     maxmw;
  double  c_const;
  double  bee;
  double  delta;

  sKernel* KER;
  int     ker_type;
  int     projection_solver, projection_projector; 
  int     PreprocessMode;
  int     preprocess_size;
  int     verbosity;
  double  tau_proximal;
  double objective_value;

// ----------------- Public Methods ---------------
  QPproblem ();
  ~QPproblem();
  int  ReadSVMFile    (char *fInput);
  int  ReadGPDTBinary(char *fName);
  int  Check2Class    (void);
  void Subproblem     (QPproblem &ker, int len, int *perm);
  void PrepMP         (void);

  double  gpdtsolve      (double *solution);
  double  pgpdtsolve     (double *solution);
  void write_solution (FILE *fp, double *sol);

// ----------------- Private Data  ---------------
private:
  int    dim;
  int    *index_in, *index_out;
  int    *ing, *inaux, *inold, *incom;
  int    *cec;
  int    nb;
  int    *bmem, *bmemrid, *pbmr;
  int    my_chunk_size;    // chunk_size for the current processor
  int    my_spD_offset;    // offset of the current processor into sp_D matrix
  int    recvl[32], displ[32];
  double kktold;
  double DELTAvpm, InitialDELTAvpm, DELTAkin;
  double *alpha;
  double *grad, *st;

// ----------------- Private Methods ---------------
private:
  int  Preprocess0 (int *aux, int *sv);
  int  Preprocess1 (sKernel* KER, int *aux, int *sv);
  int  optimal     (void);

  bool is_zero(int  i) { return (alpha[i] < DELTAsv); }
  bool is_free(int  i) 
       { return (alpha[i] > DELTAsv && alpha[i] < (c_const - DELTAsv)); }
  bool is_bound(int i) { return (alpha[i] > (c_const - DELTAsv)); }

};

void SplitParts (int n, int part, int parts, int *dim, int *off);
void SplitNum   (int n, int *nloc, int *noff);

/******************************************************************************/
/*** End of gpdt.h file                                                     ***/
/******************************************************************************/
