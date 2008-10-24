/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Konrad Rieck
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifdef HAVE_MINDY

#include <mindy.h>

#ifndef _MINDYGRAMKERNEL_H___
#define _MINDYGRAMKERNEL_H___

#include "lib/common.h"
#include "kernel/Kernel.h"

/* Similarity coefficients */
#define NO_SICO			-1

/* Parameter specifications */
typedef struct {
		char *name;									/* Name of parameter */
		int idx;										/* Index in param array (see sm.h) */
		double val;									/* Default value */
		char *descr;								/* Description */
} param_spec_t;

class CMindyGramKernel: public CKernel
{
	public:
		/* Constructors */
		CMindyGramKernel(INT ch, char *measure, DREAL width);
		CMindyGramKernel(CFeatures *l, CFeatures *r, char *measure, DREAL width);
		virtual ~CMindyGramKernel();

		/* Set options */
		void set_param(char *param);
		/* Set MD5 cache size */
		void set_md5cache(INT c);
		/* Set normalization */
		void set_norm(ENormalizationType e);

		/* Init and cleanup functions */
		void parse_params(char *);
		virtual bool init(CFeatures* l, CFeatures* r);
		virtual void cleanup();
		virtual void remove_lhs();
		virtual void remove_rhs();

		/* Identification functions */
		inline virtual EKernelType get_kernel_type() { return K_MINDYGRAM; }
		inline virtual EFeatureType get_feature_type() { return F_ULONG; }
		inline virtual EFeatureClass get_feature_class() { return C_MINDYGRAM; }
		inline virtual const char* get_name() { return "MindyGram"; }

		/* Optimization functions */
		virtual bool init_optimization(INT count, INT *IDX, DREAL * weights);
		virtual bool delete_optimization();
		virtual DREAL compute_optimized(INT idx);
		virtual void add_to_normal(INT idx, DREAL weight);
		virtual void clear_normal();

		/* Load and ysave functions */
		bool load_init(FILE* src);
		bool save_init(FILE* dest);

	protected:
		/* Kernel function */
		DREAL compute(INT idx_a, INT idx_b);

	private:
		/* Name of similartiy measure */
		char *measure;
		/* Similarity coefficient or 0 */
		sico_t simcof;
		/* Normalization mode */
		ENormalizationType norm;
		/* Kernel function (=> Mindy similarity measure) */
		sm_t *kernel;
		/* Normal vector for optimization */
		gram_t *normal;
		/* MD5 cache size */
		size_t cache;
		/* Kernel width */
		DREAL width;

};
#endif /* _MINDYGRAMKERNEL_H__ */
#endif /* HAVE_MINDY */
