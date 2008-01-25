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

#include <math.h>
#include <limits.h>

#include "base/SGObject.h"
#include "lib/io.h"
#include "lib/common.h"
#include "kernel/Kernel.h"

/** class GMNPLib */
class CGMNPLib: public CSGObject
{
	public:
		/** constructor
		 *
		 * @param vector_y vector y
		 * @param kernel kernel
		 * @param num_data number of data
		 * @param num_virtual_data number of virtual data
		 * @param num_classes number of classes
		 * @param reg_const reg const
		 */
		CGMNPLib(DREAL* vector_y, CKernel* kernel, INT num_data, INT num_virtual_data, INT num_classes, DREAL reg_const);

		~CGMNPLib();

		/** --------------------------------------------------------------
		  GMNP solver based on improved MDM algorithm 1.

		  Search strategy: u determined by common rule and v is 
		  optimized.

Usage: exitflag = gmnp_imdm( &get_col, diag_H, vector_c, dim,  
tmax, tolabs, tolrel, th, &alpha, &t, &History );
-------------------------------------------------------------- */
		int gmnp_imdm(double *vector_c,
				INT dim,
				INT tmax,
				double tolabs,
				double tolrel,
				double th,
				double *alpha,
				INT  *ptr_t,
				double **ptr_History,
				INT verb);

		/** get indices2
		 *
		 * @param index index
		 * @param c c
		 * @param i i
		 */
		void get_indices2( INT *index, INT *c, INT i );

	protected:
		/** get kernel col
		 *
		 * @param a a
		 * @return col at a
		 */
		DREAL *get_kernel_col( INT a );

		/** get col
		 *
		 * @param a a
		 * @param b b
		 * @return col at a, b
		 */
		DREAL* get_col( INT a, INT b );

		/** kernel fce
		 *
		 * @param a a
		 * @param b b
		 * @return something floaty
		 */
		double kernel_fce( INT a, INT b );

	protected:
		/** diag H */
		DREAL* diag_H;
		/** kernel columns */
		DREAL** kernel_columns;
		/** cache index */
		DREAL* cache_index;
		/** first kernel inx */
		INT first_kernel_inx;
		/** cache size */
		LONG Cache_Size;
		/** num data */
		INT m_num_data;
		/** reg const */
		DREAL m_reg_const;
		/** vectory */
		DREAL* m_vector_y;
		/** kernel */
		CKernel* m_kernel;

		/** index of first used column */
		INT first_virt_inx;
		/** cache for three columns */
		DREAL *virt_columns[3];
		/** number of virt data */
		INT m_num_virt_data;
		/** number of classes */
		INT m_num_classes;
};


#endif //GMNPLIB_H__
