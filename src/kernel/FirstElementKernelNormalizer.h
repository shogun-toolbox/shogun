/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Soeren Sonnenburg
 * Copyright (C) 2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _FIRSTELEMENTKERNELNORMALIZER_H___
#define _FIRSTELEMENTKERNELNORMALIZER_H___

#include "kernel/KernelNormalizer.h"

/** Normalize the kernel by a constant obtained from the first element of the
 * kernel matrix, i.e. \f$ c=k({\bf x},{\bf x})\f$
 *
 * \f[
 * k'(x,x')= \frac{k(x,x')}{c}
 * \f]
 *
 * useful if the kernel returns constant elements along the diagonal anyway and
 * all one wants is to scale the kernel down to 1 on the diagonal.
 */
class CFirstElementKernelNormalizer : public CKernelNormalizer
{
	public:
		/** constructor
		 */
		CFirstElementKernelNormalizer()
		{
		}

		/** default destructor */
		virtual ~CFirstElementKernelNormalizer()
		{
		}

		/** initialization of the normalizer (if needed)
         * @param k kernel */
		virtual bool init(CKernel* k)
		{
			CFeatures* old_lhs=k->lhs;
			CFeatures* old_rhs=k->rhs;
			k->lhs=old_lhs;
			k->rhs=old_lhs;

			scale=k->compute(0, 0);

			k->lhs=old_lhs;
			k->rhs=old_rhs;

			return true;
		}

		/** normalize the kernel value
		 * @param value kernel value
		 * @param idx_lhs index of left hand side vector
		 * @param idx_rhs index of right hand side vector
		 */
		inline virtual DREAL normalize(DREAL value, INT idx_lhs, INT idx_rhs)
		{
			return value/scale;
		}

		/** normalize only the left hand side vector
		 * @param value value of a component of the left hand side feature vector
		 * @param idx_lhs index of left hand side vector
		 */
		inline virtual DREAL normalize_lhs(DREAL value, INT idx_lhs)
		{
			return value/sqrt(scale);
		}

		/** normalize only the right hand side vector
		 * @param value value of a component of the right hand side feature vector
		 * @param idx_rhs index of right hand side vector
		 */
		inline virtual DREAL normalize_rhs(DREAL value, INT idx_rhs)
		{
			return value/sqrt(scale);
		}

	protected:
		/// scale constant obtained from k(0,0)
		DREAL scale;
};

#endif
