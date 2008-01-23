/*-----------------------------------------------------------------------
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Library of solvers for Generalized Nearest Point Problem (GNPP).
 *
 * Written (W) 1999-2008 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 1999-2008 Center for Machine Perception, CTU FEL Prague 
 *
-------------------------------------------------------------------- */

#ifndef GNPPLIB_H__ 
#define GNPPLIB_H__ 

#include <math.h>
#include <limits.h>

#include "base/SGObject.h"
#include "lib/io.h"
#include "lib/common.h"
#include "kernel/Kernel.h"

class CGNPPLib: public CSGObject
{
 public:
  CGNPPLib(DREAL* vector_y, CKernel* kernel, INT num_data, DREAL reg_const);

  ~CGNPPLib();

  /* --------------------------------------------------------------
     QP solver based on MDM algorithm.

     Usage: exitflag = gnpp_mdm(diag_H, vector_c, vector_y,
     dim, tmax, tolabs, tolrel, th, &alpha, &t, &aHa11, &aHa22, &History );
     -------------------------------------------------------------- */
  int gnpp_mdm(double *diag_H,
               double *vector_c,
               double *vector_y,
               INT dim, 
               INT tmax,
               double tolabs,
               double tolrel,
               double th,
               double *alpha,
               INT  *ptr_t, 
               double *ptr_aHa11,
               double *ptr_aHa22,
               double **ptr_History,
               INT verb);

  /* --------------------------------------------------------------
     QP solver based on improved MDM algorithm (u fixed v optimized)

     Usage: exitflag = gnpp_imdm( diag_H, vector_c, vector_y,
     dim, tmax, tolabs, tolrel, th, &alpha, &t, &aHa11, &aHa22, &History );
     -------------------------------------------------------------- */
  int gnpp_imdm(double *diag_H,
                double *vector_c,
                double *vector_y,
                INT dim, 
                INT tmax,
                double tolabs,
                double tolrel,
                double th,
                double *alpha,
                INT  *ptr_t, 
                double *ptr_aHa11,
                double *ptr_aHa22,
                double **ptr_History,
                INT verb);

 protected:
  DREAL* get_col( long a, long b ); 
  DREAL** kernel_columns;
  DREAL* cache_index;
  INT first_kernel_inx;
  LONG Cache_Size;
  INT m_num_data;
  DREAL m_reg_const;
  DREAL* m_vector_y;
  CKernel* m_kernel;

};

#endif // GNPPLIB_H__ 

