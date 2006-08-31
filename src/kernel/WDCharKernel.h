/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _WDCHARKERNEL_H___
#define _WDCHARKERNEL_H___

#include "lib/common.h"
#include "kernel/CharKernel.h"
#include "kernel/WeightedDegreeCharKernel.h"

enum EWDKernType
{
	E_WD=0,
	E_CONST=1,
	E_LINEAR=2,
	E_SQPOLY=3,
	E_CUBICPOLY=4,
	E_EXP=5,
	E_LOG=6,
	E_EXTERNAL=7
};

class CWDCharKernel: public CCharKernel
{
	public:
		CWDCharKernel(LONG size, EWDKernType type, INT degree, INT which_deg=-1, bool use_normalization=true);
		~CWDCharKernel();

		virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
		virtual void cleanup();

		/// load and save kernel init_data
		bool load_init(FILE* src);
		bool save_init(FILE* dest);

		/// set parms
		virtual bool set_kernel_parameters(INT num, const double* param);

		// return what type of kernel we are Linear,Polynomial, Gaussian,...
		virtual EKernelType get_kernel_type() { return K_WEIGHTEDDEGREE; }

		// return the name of a kernel
		virtual const CHAR* get_name() { return "WD"; }
	protected:
		bool init_matching_weights();

		bool init_matching_weights_wd();
		bool init_matching_weights_const();
		bool init_matching_weights_linear();
		bool init_matching_weights_sqpoly();
		bool init_matching_weights_cubicpoly();
		bool init_matching_weights_exp();
		bool init_matching_weights_log();
		bool init_matching_weights_external();

		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		DREAL compute(INT idx_a, INT idx_b);
		/*    compute_kernel*/

		virtual void remove_lhs();
		virtual void remove_rhs();

	protected:
		INT num_matching_weights_external;
		DREAL* matching_weights_external;

		DREAL* matching_weights;

		EWDKernType type;
		INT which_degree;
		INT degree;
		INT seq_length;

		double* sqrtdiag_lhs;
		double* sqrtdiag_rhs;

		bool use_normalization;
		bool initialized;
		bool* match_vector;
};
#endif
