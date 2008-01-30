/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _AUCKERNEL_H___
#define _AUCKERNEL_H___

#include "lib/common.h"
#include "kernel/SimpleKernel.h"
#include "features/WordFeatures.h"

/** kernel AUC */
class CAUCKernel: public CSimpleKernel<WORD>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param subkernel the subkernel
		 */
		CAUCKernel(INT size, CKernel* subkernel);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param subkernel the subkernel
		 */
		CAUCKernel(CWordFeatures *l, CWordFeatures *r, CKernel* subkernel);

		virtual ~CAUCKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** clean up kernel */
		virtual void cleanup();

		/** load kernel init_data
		 *
		 * @param src file to load from
		 * @return if loading was successful
		 */
		virtual bool load_init(FILE* src);

		/** save kernel init_data
		 *
		 * @param dest file to save to
		 * @return if saving was successful
		 */
		virtual bool save_init(FILE* dest);

		/** return what type of kernel we are
		 *
		 * @return kernel type AUC
		 */
		virtual EKernelType get_kernel_type() { return K_AUC; }

		/** return the kernel's name
		 *
		 * @return name AUC
		 */
		virtual const CHAR* get_name() { return "AUC" ; }

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual DREAL compute(INT idx_a, INT idx_b);

	protected:
		/** the subkernel */
		CKernel* subkernel;
};

#endif /* _AUCKERNEL_H__ */
