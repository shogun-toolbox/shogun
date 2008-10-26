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

/** class GNPPLib */
class CGNPPLib: public CSGObject
{
 public:
  /** constructor
   *
   * @param vector_y vector y
   * @param kernel kernel
   * @param num_data number of data
   * @param reg_const reg const
   */
  CGNPPLib(DREAL* vector_y, CKernel* kernel, int32_t num_data, DREAL reg_const);
  ~CGNPPLib();

  /** --------------------------------------------------------------
     QP solver based on MDM algorithm.

     Usage: exitflag = gnpp_mdm(diag_H, vector_c, vector_y,
     dim, tmax, tolabs, tolrel, th, &alpha, &t, &aHa11, &aHa22, &History );
     -------------------------------------------------------------- */
  int8_t gnpp_mdm(double *diag_H,
               double *vector_c,
               double *vector_y,
               int32_t dim,
               int32_t tmax,
               double tolabs,
               double tolrel,
               double th,
               double *alpha,
               int32_t  *ptr_t,
               double *ptr_aHa11,
               double *ptr_aHa22,
               double **ptr_History,
               int32_t verb);

  /** --------------------------------------------------------------
     QP solver based on improved MDM algorithm (u fixed v optimized)

     Usage: exitflag = gnpp_imdm( diag_H, vector_c, vector_y,
     dim, tmax, tolabs, tolrel, th, &alpha, &t, &aHa11, &aHa22, &History );
     -------------------------------------------------------------- */
  int8_t gnpp_imdm(double *diag_H,
                double *vector_c,
                double *vector_y,
                int32_t dim, 
                int32_t tmax,
                double tolabs,
                double tolrel,
                double th,
                double *alpha,
                int32_t  *ptr_t, 
                double *ptr_aHa11,
                double *ptr_aHa22,
                double **ptr_History,
                int32_t verb);

 protected:
  /** get col
   *
   * @param a a
   * @param b b
   * @return something floaty
   */
  DREAL* get_col( long a, long b );

  /** kernel columns */
  DREAL** kernel_columns;
  /** cache index */
  DREAL* cache_index;
  /** first kernel inx */
  int32_t first_kernel_inx;
  /** cache size */
  int64_t Cache_Size;
  /** num data */
  int32_t m_num_data;
  /** reg const */
  DREAL m_reg_const;
  /** vector y */
  DREAL* m_vector_y;
  /** kernel */
  CKernel* m_kernel;

};

#endif // GNPPLIB_H__ 

