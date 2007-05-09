/*-----------------------------------------------------------------------
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Library for solving QP task required for learning SVM without bias term. 
 *
 * Written (W) 1999-2007 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 1999-2007 Center for Machine Perception, CTU FEL Prague 
 *
-------------------------------------------------------------------- */

#ifndef QPBSVMLIB_H__ 
#define QPBSVMLIB_H__ 

#include <math.h>
#include <limits.h>

#include "base/SGObject.h"
#include "lib/io.h"
#include "lib/common.h"
#include "kernel/Kernel.h"

class CQPBSVMLib: public CSGObject
{
 public:
  CQPBSVMLib(DREAL* vector_y, CKernel* kernel, INT num_data, INT num_virtual_data, INT num_classes, DREAL reg_const);

  ~CQPBSVMLib();

/* --------------------------------------------------------------

Usage: exitflag = qpbsvm_sca( &get_col, diag_H, f, UB, dim, tmax, 
               tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb )

-------------------------------------------------------------- */
int qpbsvm_sca(const void* (*get_col)(long,long),
            double *diag_H,
            double *f,
            double UB,
            long   dim,
            long   tmax,
            double tolabs,
            double tolrel,
            double tolKKT,
            double *x,
	        double *Nabla,
            long   *ptr_t,
            double **ptr_History,
            long   verb);
/* --------------------------------------------------------------

Usage: exitflag = qpbsvm_scas( &get_col, diag_H, f, UB, dim, tmax, 
               tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb )

-------------------------------------------------------------- */
int qpbsvm_scas(const void* (*get_col)(long,long),
            double *diag_H,
            double *f,
            double UB,
            long   dim,
            long   tmax,
            double tolabs,
            double tolrel,
            double tolKKT,
            double *x,
	        double *Nabla,
            long   *ptr_t,
            double **ptr_History,
            long   verb);

/* --------------------------------------------------------------

Usage: exitflag = qpbsvm_scamv( &get_col, diag_H, f, UB, dim, tmax, 
               tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb )

-------------------------------------------------------------- */
int qpbsvm_scamv(const void* (*get_col)(long,long),
            double *diag_H,
            double *f,
            double UB,
            long   dim,
            long   tmax,
            double tolabs,
            double tolrel,
            double tolKKT,
            double *x,
	        double *Nabla,
            long   *ptr_t,
            double **ptr_History,
            long   verb);
};


#endif //QPBSVMLIB_H__
