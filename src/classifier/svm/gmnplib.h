/*-----------------------------------------------------------------------
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Library of solvers for Generalized Nearest Point Problem (GNPP).
 *
 * Written (W) 1999-2007 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 1999-2007 Center for Machine Perception, CTU FEL Prague 
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

class CGMNPLib: public CSGObject
{
 public:
  CGMNPLib(DREAL* vector_y, CKernel* kernel, INT num_data, INT num_virtual_data, INT num_classes, DREAL reg_const);

  ~CGMNPLib();

/* --------------------------------------------------------------
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

 protected:
	void get_indices2( INT *index, INT *c, INT i );
	DREAL *get_kernel_col( INT a );
	DREAL* get_col( INT a, INT b ); 
	double kernel_fce( INT a, INT b );

 protected:
	DREAL* diag_H;
	DREAL** kernel_columns;
	DREAL* cache_index;
	INT first_kernel_inx;
	LONG Cache_Size;
	INT m_num_data;
	DREAL m_reg_const;
	DREAL* m_vector_y;
	CKernel* m_kernel;

	INT first_virt_inx;                 /* index of first used column */
	DREAL *virt_columns[3];            /* cache for three columns*/
	INT m_num_virt_data;
	INT m_num_classes;
};


#endif //GMNPLIB_H__
