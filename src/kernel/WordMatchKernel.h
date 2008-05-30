/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _WORDMATCHKERNEL_H___
#define _WORDMATCHKERNEL_H___

#include "lib/common.h"
#include "kernel/SimpleKernel.h"
#include "features/WordFeatures.h"

/** kernel WordMatch */
class CWordMatchKernel: public CSimpleKernel<WORD>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param d degree
		 * @param do_rescale if rescaling shall be applied
		 * @param scale scaling factor
		 */
		CWordMatchKernel(INT size, INT d,
			bool do_rescale=true, DREAL scale=1.);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param degree degree
		 * @param do_rescale if rescaling shall be applied
		 * @param scale scaling factor
		 */
		CWordMatchKernel(CWordFeatures* l, CWordFeatures* r, INT degree,
			bool do_rescale=true, DREAL scale=1.);

		virtual ~CWordMatchKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

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
		 * @return kernel type LINEAR
		 */
		virtual EKernelType get_kernel_type() { return K_LINEAR; }

		/** return the kernel's name
		 *
		 * @return name WordMatch
		 */
		virtual const CHAR* get_name() { return "WordMatch"; }

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

		/** initialize rescaling */
		virtual void init_rescale();

	protected:
		/** scaling factor */
		double scale;
		/** if rescaling shall be applied */
		bool do_rescale;
		/** if kernel is initialized */
		bool initialized;

		/** degree */
		INT degree;
};

#endif /* _WORDMATCHKERNEL_H__ */
