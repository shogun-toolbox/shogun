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

#ifndef GMNPLIB_H__
#define GMNPLIB_H__

#include <shogun/mathematics/Math.h>
#include <limits.h>

#include <shogun/base/SGObject.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{
/** @brief class GMNPLib
 * Library of solvers for Generalized Minimal Norm Problem (GMNP).
 *
 *  Generalized Minimal Norm Problem to solve is
 *
 *   min 0.5*alpha'*H*alpha + c'*alpha
 *
 *   subject to  sum(alpha) = 1,  alpha(i) >= 0
 *
 *  H [dim x dim] is symmetric positive definite matrix.
 *  c [dim x 1] is an arbitrary vector.
 *
 *  The precision of the found solution is given by
 *  the parameters tmax, tolabs and tolrel which
 *  define the stopping conditions:
 *
 *  UB-LB <= tolabs      ->  exit_flag = 1   Abs. tolerance.
 *  UB-LB <= UB*tolrel   ->  exit_flag = 2   Relative tolerance.
 *  LB > th              ->  exit_flag = 3   Threshold on lower bound.
 *  t >= tmax            ->  exit_flag = 0   Number of iterations.
 *
 *  UB ... Upper bound on the optimal solution.
 *  LB ... Lower bound on the optimal solution.
 *  t  ... Number of iterations.
 *  History ... Value of LB and UB wrt. number of iterations.
 *
 *
 *  The following algorithms are implemented:
 *  ..............................................
 *
 *  - GMNP solver based on improved MDM algorithm 1 (u fixed v optimized)
 *     exitflag = gmnp_imdm( &get_col, diag_H, vector_c, dim,
 *                  tmax, tolabs, tolrel, th, &alpha, &t, &History, verb  );
 *
 *   For more info refer to V.Franc: Optimization Algorithms for Kernel
 *   Methods. Research report. CTU-CMP-2005-22. CTU FEL Prague. 2005.
 *   ftp://cmp.felk.cvut.cz/pub/cmp/articles/franc/Franc-PhD.pdf .
*/
class CGMNPLib: public CSGObject
{
	public:
		/** default constructor  */
		CGMNPLib();

		/** constructor
		 *
		 * @param vector_y vector y
		 * @param kernel kernel
		 * @param num_data number of data
		 * @param num_virtual_data number of virtual data
		 * @param num_classes number of classes
		 * @param reg_const reg const
		 */
		CGMNPLib(
			float64_t* vector_y, CKernel* kernel, int32_t num_data,
			int32_t num_virtual_data, int32_t num_classes, float64_t reg_const);

		virtual ~CGMNPLib();

		/** --------------------------------------------------------------
		  GMNP solver based on improved MDM algorithm 1.

		  Search strategy: u determined by common rule and v is
		  optimized.

Usage: exitflag = gmnp_imdm( &get_col, diag_H, vector_c, dim,
tmax, tolabs, tolrel, th, &alpha, &t, &History );
-------------------------------------------------------------- */
		int8_t gmnp_imdm(float64_t *vector_c,
				int32_t dim,
				int32_t tmax,
				float64_t tolabs,
				float64_t tolrel,
				float64_t th,
				float64_t *alpha,
				int32_t  *ptr_t,
				float64_t **ptr_History,
				int32_t verb);

		/** get indices2
		 *
		 * @param index index
		 * @param c c
		 * @param i i
		 */
		void get_indices2( int32_t *index, int32_t *c, int32_t i );

	protected:
		/** get kernel col
		 *
		 * @param a a
		 * @return col at a
		 */
		float64_t *get_kernel_col( int32_t a );

		/** get col
		 *
		 * @param a a
		 * @param b b
		 * @return col at a, b
		 */
		float64_t* get_col( int32_t a, int32_t b );

		/** kernel fce
		 *
		 * @param a a
		 * @param b b
		 * @return something floaty
		 */
		float64_t kernel_fce( int32_t a, int32_t b );

		/** @return object name */
		virtual const char* get_name() const { return "GMNPLib"; }

	protected:
		/** diag H */
		float64_t* diag_H;
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
		/** vectory */
		float64_t* m_vector_y;
		/** kernel */
		CKernel* m_kernel;

		/** index of first used column */
		int32_t first_virt_inx;
		/** cache for three columns */
		float64_t *virt_columns[3];
		/** number of virt data */
		int32_t m_num_virt_data;
		/** number of classes */
		int32_t m_num_classes;
};
}
#endif //GMNPLIB_H__
