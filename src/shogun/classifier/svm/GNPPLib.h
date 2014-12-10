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

#include <limits.h>

#include <shogun/mathematics/Math.h>
#include <shogun/base/SGObject.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{
/** @brief class GNPPLib, a Library of solvers for Generalized Nearest Point
 * Problem (GNPP).
 */
class CGNPPLib: public CSGObject
{
 public:
  /** default constructor  */
  CGNPPLib();

  /** constructor
   *
   * @param vector_y vector y
   * @param kernel kernel
   * @param num_data number of data
   * @param reg_const reg const
   */
  CGNPPLib(float64_t* vector_y, CKernel* kernel, int32_t num_data, float64_t reg_const);
  virtual ~CGNPPLib();

  /** --------------------------------------------------------------
     QP solver based on MDM algorithm.

     Usage: exitflag = gnpp_mdm(diag_H, vector_c, vector_y,
     dim, tmax, tolabs, tolrel, th, &alpha, &t, &aHa11, &aHa22, &History );
     -------------------------------------------------------------- */
  int8_t gnpp_mdm(float64_t *diag_H,
               float64_t *vector_c,
               float64_t *vector_y,
               int32_t dim,
               int32_t tmax,
               float64_t tolabs,
               float64_t tolrel,
               float64_t th,
               float64_t *alpha,
               int32_t  *ptr_t,
               float64_t *ptr_aHa11,
               float64_t *ptr_aHa22,
               float64_t **ptr_History,
               int32_t verb);

  /** --------------------------------------------------------------
     QP solver based on improved MDM algorithm (u fixed v optimized)

     Usage: exitflag = gnpp_imdm( diag_H, vector_c, vector_y,
     dim, tmax, tolabs, tolrel, th, &alpha, &t, &aHa11, &aHa22, &History );
     -------------------------------------------------------------- */
  int8_t gnpp_imdm(float64_t *diag_H,
                float64_t *vector_c,
                float64_t *vector_y,
                int32_t dim, 
                int32_t tmax,
                float64_t tolabs,
                float64_t tolrel,
                float64_t th,
                float64_t *alpha,
                int32_t  *ptr_t, 
                float64_t *ptr_aHa11,
                float64_t *ptr_aHa22,
                float64_t **ptr_History,
                int32_t verb);

  /** @return object name */
  virtual const char* get_name() const { return "GNPPLib"; }

 protected:
  /** get col
   *
   * @param a a
   * @param b b
   * @return something floaty
   */
  float64_t* get_col(int64_t a, int64_t b);

  /** kernel columns */
  float64_t** kernel_columns;
  /** cache index */
  float64_t* cache_index;
  /** first kernel inx */
  int32_t first_kernel_inx;
  /** cache size */
  int64_t Cache_Size;
  /** num data */
  int32_t m_num_data;
  /** reg const */
  float64_t m_reg_const;
  /** vector y */
  float64_t* m_vector_y;
  /** kernel */
  CKernel* m_kernel;

};
}
#endif // GNPPLIB_H__ 
