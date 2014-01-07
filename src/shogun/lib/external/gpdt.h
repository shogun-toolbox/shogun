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
 *** File:     gpdt.h                                                       ***
 *** Type:     scalar                                                       ***
 *** Version:  1.0                                                          ***
 *** Date:     October, 2005                                                ***
 *** Revision: 1                                                            ***
 ***                                                                        ***
 ******************************************************************************/
#include <kernel/Kernel.h>

#ifndef DOXYGEN_SHOULD_SKIP_THIS

namespace shogun
{
#define MAXLENGTH 256
#define cachetype KERNELCACHE_ELEM
#define EPS_SV    1.0e-9   /* precision for multipliers */

enum {
  SOLVER_VPM      = 0,
  SOLVER_FLETCHER = 1
};

/** s kernel */
class sKernel
{
public:
  /** kernel type */
  int32_t  ker_type;
  /** lx */
  int32_t  *lx;
  /** ix */
  int32_t  **ix;
  /** x */
  float32_t  **x;
  /** nor */
  float64_t *nor;
  /** sigma */
  float64_t sigma;
  /** degree */
  float64_t degree;
  /** normalization factor */
  float64_t norm;
  /** c poly */
  float64_t c_poly;
  /** kernel evaluations */
  float64_t KernelEvaluations;

  /** call kernel fun
   *
   * @param i
   * @param j
   * @return something floaty
   */
  float64_t (sKernel::*kernel_fun)(int32_t i, int32_t j);

  /** constructor
   *
   * @param k kernel
   * @param ell ell
   */
  sKernel (shogun::CKernel* k, int32_t ell);
  ~sKernel();

  /** set data
   *
   * @param x_ new x
   * @param ix_ new ix
   * @param lx_ new lx
   * @param ell new ell
   * @param dim dim
   */
  void SetData(
	float32_t **x_, int32_t **ix_, int32_t *lx_, int32_t ell, int32_t dim);

  /** set subproblem
   *
   * @param ker kernel
   * @param len len
   * @param perm perm
   */
  void   SetSubproblem (sKernel* ker, int32_t len, int32_t *perm);

  /** get an item from the kernel
   *
   * @param i index i
   * @param j index j
   * @return item from kernel at index i, j
   */
  float64_t Get(int32_t i, int32_t j)
  {
    KernelEvaluations += 1.0F;
    return kernel->kernel(i, j);
  }

  /** add something
   *
   * @param v v
   * @param j j
   * @param mul mul
   */
  void   Add           (float64_t *v, int32_t j, float64_t mul);

  /** prod something
   *
   * @param v v
   * @param j j
   * @return something floaty
   */
  float64_t Prod          (float64_t *v, int32_t j);

  /** get kernel
   *
   * @return kernel
   */
  inline CKernel* get_kernel()
  {
    return kernel;
  }

private:
  CKernel* kernel;
  int32_t    vauxRow;
  int32_t    IsSubproblem;
  int32_t    ell, dim;
  float32_t  *vaux;

  float64_t dot     (int32_t i, int32_t j);
};

void SplitParts (
	int32_t n, int32_t part, int32_t parts, int32_t *dim, int32_t *off);
void SplitNum   (int32_t n, int32_t *nloc, int32_t *noff);
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

/******************************************************************************/
/*** End of gpdt.h file                                                     ***/
/******************************************************************************/
